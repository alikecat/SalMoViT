from pathlib import Path
import time

import numpy as np
import torch
from rich import print
from rich.progress import Progress
from utils import sparse_dense_elem_mul
from torchvision.transforms import Normalize
from config import DEEPGAZE_PATH


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device=torch.device("cpu"),
        quick_validation=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.quick_validation = quick_validation
        self.normalizer = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def train_step(self, images, coords):
        pt_counts = [len(coord[0]) for coord in coords]
        batch_indices = np.repeat(np.arange(len(pt_counts)), pt_counts)
        all_coords = np.concatenate([np.transpose(coord) for coord in coords])
        output = self.model(images.to(self.device))
        loss = -output[batch_indices, :, all_coords[:, 0], all_coords[:, 1]].mean()
        return loss

    def valid_step(
        self, images, shapes, coords, fixation_maps, gauss_filter, quick_valid=False
    ):
        valid_indices = [i for i, coord in enumerate(coords) if coord.size > 0]
        if not valid_indices:
            return None
        images = images[valid_indices]
        shapes, coords = [[x[i] for i in valid_indices] for x in (shapes, coords)]
        metrics = torch.zeros(len(coords), 7 if fixation_maps else 2)
        if self.model._get_name() == "DeepGazeIIE":
            centerbias = torch.tensor(
                np.load(Path(DEEPGAZE_PATH) / "centerbias_mit1003.npy")
            ).to(self.device)[None, None, :, :]
            centerbias = torch.nn.functional.interpolate(
                centerbias, size=shapes[0][:2], mode="nearest"
            )
            centerbias -= torch.logsumexp(centerbias, dim=(2, 3))
            output = self.model(
                images.to(self.device) * 255, centerbias=centerbias.squeeze(0)
            ).squeeze(1)
        elif self.model._get_name() == "UNISAL":
            ar = shapes[0][0] / shapes[0][1]
            min_prod = 100
            max_prod = 120
            ar_array = []
            size_array = []
            for n1 in range(7, 14):
                for n2 in range(7, 14):
                    if min_prod <= n1 * n2 <= max_prod:
                        this_ar = n1 / n2
                        ar_array.append(min((ar, this_ar)) / max((ar, this_ar)))
                        size_array.append((n1, n2))
            bn_size = size_array[np.argmax(np.array(ar_array)).item()]
            out_size = tuple(r * 32 for r in bn_size)
            images = torch.nn.functional.interpolate(
                images, size=out_size, mode="bilinear", align_corners=False
            )
            images = self.normalizer(images)
            output = self.model(images.to(self.device).unsqueeze(0), static=True)
            output = torch.nn.functional.interpolate(
                output.squeeze(1),
                size=shapes[0][:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            output = self.model(images.to(self.device)).squeeze(1)
        output_area = torch.tensor(output.shape[-2:]).prod()
        output -= torch.logsumexp(output, dim=(1, 2), keepdim=True)
        den_output = torch.exp(output)
        den_output = den_output / den_output.sum((1, 2), keepdim=True)
        norm_output = (den_output - 1 / output_area) / den_output.std(
            (1, 2), keepdim=True
        )
        current_fixation_maps = [
            torch.sparse_coo_tensor(
                coord,
                torch.ones(len(coord[0]), dtype=torch.long),
                shape[:2],
            ).coalesce()
            for shape, coord in zip(shapes, coords)
        ]
        if not quick_valid:
            other_fixation_map = [
                (fixation_maps[tuple(shapes[0][:2])] - cf_map).coalesce()
                for cf_map in current_fixation_maps
            ]
            gauss_map = gauss_filter(
                torch.stack(current_fixation_maps)
                .to(self.device)
                .to_dense()
                .unsqueeze(1)
                .float()
            ).squeeze(1)
            norm_gauss_map = (
                gauss_map - gauss_map.mean((1, 2), keepdim=True)
            ) / gauss_map.std((1, 2), keepdim=True)
            den_gauss_map = gauss_map / gauss_map.sum((1, 2), keepdim=True)
        for i, coord in enumerate(coords):
            output_element = output[i, *coord]
            ig = (
                (output_element + torch.log(output_area)) / torch.log(torch.tensor(2))
            ).mean()
            nss = norm_output[i, *coord].mean()
            if quick_valid:
                auc = sauc = cc = kld = sim = torch.tensor(
                    torch.nan, device=self.device
                )
            else:
                auc_sum, sauc_sum = 0, 0
                for elem in output_element:
                    less = output[i] < elem
                    equal = output[i] == elem
                    auc_sum += (less.sum() + equal.sum() / 2) / output_area
                    fixation = other_fixation_map[i].to(self.device)
                    sauc_sum += (
                        sparse_dense_elem_mul(fixation, less).sum()
                        + sparse_dense_elem_mul(fixation, equal).sum() / 2
                    ) / (fixation.sum())
                auc = auc_sum / len(output_element)
                sauc = (sauc_sum / len(output_element)).mean()
                cc = (
                    torch.tensor(0, device=self.device)
                    if torch.all(output[i] == output[i].flatten()[0])
                    else torch.corrcoef(
                        torch.stack(
                            [norm_gauss_map[i].flatten(), norm_output[i].flatten()]
                        )
                    )[0, 1]
                )
                kld = (
                    den_gauss_map[i]
                    * torch.log(
                        den_gauss_map[i]
                        / (den_output[i] + torch.finfo(torch.float64).eps)
                        + torch.finfo(torch.float64).eps
                    )
                ).sum()
                sim = torch.min(den_output[i], den_gauss_map[i]).sum()
            metrics[i] = torch.stack([ig, nss, auc, sauc, cc, kld, sim])
        return metrics.detach().cpu().numpy()

    def train_epoch(self, train_loader, desc=""):
        learning_rate = self.optimizer.param_groups[0]["lr"]
        with Progress(transient=True) as progress:
            task = progress.add_task(
                desc + f" & LR={learning_rate:.3g}", total=len(train_loader)
            )
            for images, _, coords in train_loader:
                self.optimizer.zero_grad()
                loss = self.train_step(images, coords)
                loss.backward()
                self.optimizer.step()
                progress.update(task, advance=1)
        duration = time.strftime("%H:%M:%S", time.gmtime(progress.tasks[task].elapsed))
        print(f"{desc} {duration}, LR={learning_rate:.3g}")

    def valid_epoch(self, valid_component, desc, quick_valid=False):
        with Progress(transient=True) as progress:
            task = progress.add_task(desc, total=len(valid_component[0]))
            metrics = []
            for images, shapes, coords in valid_component[0]:
                metrics.append(
                    self.valid_step(
                        images, shapes, coords, *valid_component[1:], quick_valid
                    )
                )
                progress.update(task, advance=1)
            duration = time.strftime(
                "%H:%M:%S", time.gmtime(progress.tasks[task].elapsed)
            )
            metrics = np.concatenate(metrics).mean(0).tolist()
            metric_names = ["IG", "NSS", "AUC", "SAUC", "CC", "KLD", "SIM"]
            metrics_str = ", ".join(
                [f"{k.upper()}={v:.3f}" for k, v in zip(metric_names, metrics)]
            )
            print(f"{desc}{duration}, NUM={len(valid_component[0]):>4}, {metrics_str}")
        return metrics
