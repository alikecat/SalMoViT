import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, List

import numpy as np
import pandas as pd
import torch
import typer
from config import DatasetConfig, ModelConfig, SchedulerType, DEEPGAZE_PATH
from data_utils import SaliencyDataset, split_component_collate
from salmovit_model import SalMoViT
from rich import print, table
from torch.utils.data import ConcatDataset, DataLoader
from trainer import Trainer
from utils import GaussFilter, precall_param_merge

app = typer.Typer(help="SalMoViT Saliency Prediction", no_args_is_help=True)


@dataclass
class HyperConfigs:
    command: str = None
    dataset_configs: list[DatasetConfig] = None
    train_dataset_configs: list[DatasetConfig] = None
    valid_dataset_configs: list[DatasetConfig] = None
    num_folds: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = int(hash(str(time.time())) % (2**32))
    debug: bool = False
    weight_dir: Path = Path(__file__).parent.joinpath("Weights")
    result_dir: Path = Path(__file__).parent.joinpath("Results")
    learning_rate: float = 1e-3
    backbone_lr_factor: float = 0.2
    scheduler_type: SchedulerType = SchedulerType.plateau
    batch_size: int = 1
    epochs: int = 30
    valid_batch_size: int = 1
    validate_all_epochs: bool = False
    quick_valid: bool = False
    start_time: str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    save_time: str = None


def _prepare_dataset(cfg: HyperConfigs, ds_cfgs, dataset_table):
    datasets = []
    for ds_cfg in ds_cfgs:
        stimuli, fixations = ds_cfg.loader()
        stimuli.cached = False
        dataset = SaliencyDataset(stimuli, fixations)
        if cfg.debug and len(dataset) > 100:
            dataset = dataset[random.sample(range(len(dataset)), 100)]
        datasets.append(dataset)
        dataset_table.add_row(
            ds_cfg.name,
            str(len(stimuli)),
            str(len(fixations)),
            str(fixations.subject_count),
        )
    return datasets


def prepare_datasets(cfg: HyperConfigs):
    dataset_table = table.Table(title="Dataset Statistics", header_style="bold yellow")
    dataset_table.add_column("Dataset Name", style="bold green")
    dataset_table.add_column("Img", justify="right", style="cyan")
    dataset_table.add_column("Fix", justify="right", style="cyan")
    dataset_table.add_column("Subj", justify="right", style="cyan")
    split_datasets = []
    if cfg.dataset_configs:
        split_datasets = _prepare_dataset(cfg, cfg.dataset_configs, dataset_table)
    if cfg.command == "kfold_cross_validate":
        print(dataset_table)
        return split_datasets
    else:
        if split_datasets:
            split = [dataset.split() for dataset in split_datasets]
            train_datasets, valid_datasets = map(list, zip(*split))
        else:
            train_datasets, valid_datasets = [], []
        if cfg.train_dataset_configs:
            dataset_table.add_section()
            train_datasets += _prepare_dataset(
                cfg, cfg.train_dataset_configs, dataset_table
            )
        if cfg.valid_dataset_configs:
            dataset_table.add_section()
            valid_datasets += _prepare_dataset(
                cfg, cfg.valid_dataset_configs, dataset_table
            )
        print(dataset_table)
        return train_datasets, valid_datasets


def setup_config(
    ctx: typer.Context,
    dataset_configs: Annotated[
        List[DatasetConfig],
        typer.Option(
            "--dataset",
            "-f",
            help="Auto-split datasets (use -f before each name)",
            show_default=False,
            show_choices=False,
            case_sensitive=False,
        ),
    ] = [],
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Computation device (e.g. 'cpu' or 'cuda')"
        ),
    ] = HyperConfigs.device,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for result reproducibility")
    ] = HyperConfigs.seed,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            "-g",
            help="Enable debug mode (uses partial data)",
        ),
    ] = HyperConfigs.debug,
    weight_dir: Annotated[
        Path,
        typer.Option(
            "--weight-dir",
            "-W",
            help="Directory to save trained model weights",
            rich_help_panel="Paths",
        ),
    ] = HyperConfigs.weight_dir,
    result_dir: Annotated[
        Path,
        typer.Option(
            "--result-dir",
            "-R",
            help="Directory to save validation results",
            rich_help_panel="Paths",
        ),
    ] = HyperConfigs.result_dir,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            "-l",
            help="Learning rate",
            rich_help_panel="Training Options",
        ),
    ] = HyperConfigs.learning_rate,
    backbone_lr_factor: Annotated[
        float,
        typer.Option(
            "--backbone-lr-factor",
            "-B",
            help="Learning rate factor for backbone parameters",
            rich_help_panel="Training Options",
        ),
    ] = HyperConfigs.backbone_lr_factor,
    scheduler_type: Annotated[
        SchedulerType,
        typer.Option(
            "--scheduler",
            "-r",
            help="Learning rate scheduler type",
            rich_help_panel="Training Options",
        ),
    ] = HyperConfigs.scheduler_type,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
            rich_help_panel="Training Options",
        ),
    ] = HyperConfigs.batch_size,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs",
            rich_help_panel="Training Options",
        ),
    ] = HyperConfigs.epochs,
    valid_batch_size: Annotated[
        int,
        typer.Option(
            "--valid-batch-size",
            "-u",
            help="Validation batch size",
            rich_help_panel="Validation Options",
        ),
    ] = HyperConfigs.valid_batch_size,
    validate_all_epochs: Annotated[
        bool,
        typer.Option(
            "--valid-all-epochs",
            "-a",
            help="valid after each epoch",
            show_default="only last epoch",
            rich_help_panel="Validation Options",
        ),
    ] = HyperConfigs.validate_all_epochs,
    quick_valid: Annotated[
        bool,
        typer.Option(
            "--quick-valid",
            "-q",
            help="valid only IG/NSS",
            show_default="IG/NSS/AUC/SAUC/CC/KLD/SIM",
            rich_help_panel="Validation Options",
        ),
    ] = HyperConfigs.quick_valid,
):
    if debug:
        print("Debug mode enabled")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    valid_batch_size = valid_batch_size or batch_size
    ctx.obj = {
        "cfg": HyperConfigs(
            command=ctx.info_name,
            dataset_configs=dataset_configs,
            device=device,
            seed=seed,
            debug=debug,
            weight_dir=weight_dir,
            result_dir=result_dir,
            learning_rate=learning_rate,
            backbone_lr_factor=backbone_lr_factor,
            scheduler_type=scheduler_type,
            batch_size=batch_size,
            epochs=epochs,
            valid_batch_size=valid_batch_size,
            validate_all_epochs=validate_all_epochs,
            quick_valid=quick_valid,
        )
    }


def _create_trainer(cfg: HyperConfigs, model=None):
    model = model or SalMoViT()
    model = model.to(cfg.device)
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "encoder" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.learning_rate},
            {
                "params": backbone_params,
                "lr": cfg.learning_rate * cfg.backbone_lr_factor,
            },
        ]
    )
    if cfg.scheduler_type == SchedulerType.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
    elif cfg.scheduler_type == SchedulerType.exponential:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    elif cfg.scheduler_type == SchedulerType.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=0, factor=0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0)
    return Trainer(model, optimizer, scheduler, cfg.device, cfg.quick_valid)


def _prepare_data_loader(cfg: HyperConfigs, dataset, is_validation=False):
    return DataLoader(
        dataset,
        batch_size=cfg.valid_batch_size if is_validation else cfg.batch_size,
        shuffle=not is_validation,
        collate_fn=split_component_collate,
        pin_memory=cfg.device == "cuda",
        num_workers=4,
        prefetch_factor=2,
    )


def _save_results(cfg: HyperConfigs, valid_scores):
    index_columns = ["EPOCH", "LR", "DATASET"]
    if cfg.command == "cv":
        index_columns = ["FOLD", *index_columns]
    metric_names = ["NUM", "IG", "NSS", "AUC", "SAUC", "CC", "KLD", "SIM"]
    columns = [*index_columns, *metric_names][: len(valid_scores[0])]
    result = pd.DataFrame(valid_scores, columns=columns)
    cfg.save_time = time.strftime("%Y%m%d_%H%M%S")
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.result_dir.joinpath(f"{cfg.command}_{cfg.start_time}")
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(asdict(cfg), f, indent=4, default=str)
    result.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"💾 Saved result to {output_path.with_suffix('.csv')}")


def _run_training_loop(
    cfg: HyperConfigs, train_loader, valid_components, results, fold_idx=None
):
    trainer = _create_trainer(cfg)
    best_info = None
    max_name_len = np.max([len(name) for name in valid_components.keys()])
    early_stop_counter = 0
    for epoch in range(cfg.epochs):
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.model.train()
        ig = trainer.train_epoch(
            train_loader, f"🔥 Epoch {epoch + 1:{len(str(cfg.epochs))}d}/{cfg.epochs}"
        )
        score = [epoch + 1, lr, "train", len(train_loader), ig] + [None] * 6
        epoch_scores = [[fold_idx, *score]] if fold_idx else [score]
        if valid_components and (cfg.validate_all_epochs or epoch == cfg.epochs - 1):
            with torch.no_grad():
                trainer.model.eval()
                for name, valid_component in valid_components.items():
                    desc = f"✅ Epoch {epoch + 1:{len(str(cfg.epochs))}d}/{cfg.epochs} @ {name:<{max_name_len}}"
                    metrics = trainer.valid_epoch(
                        valid_component, desc, cfg.quick_valid
                    )
                    num = len(valid_component[0])
                    score = [epoch + 1, lr, f"val_{name}", num, *metrics]
                    epoch_scores.append(score)
                if len(valid_components) > 1:
                    num = sum([score[3] for score in epoch_scores[1:]])
                    metric_names = ["IG", "NSS", "AUC", "SAUC", "CC", "KLD", "SIM"]
                    metrics = np.mean([score[4:] for score in epoch_scores[1:]], 0)
                    score = [epoch + 1, lr, "val_DS-Avg", num, *metrics]
                    epoch_scores.append(score)
                    metric_names = ["IG", "NSS", "AUC", "SAUC", "CC", "KLD", "SIM"]
                    metrics_str = ", ".join(
                        [f"{k.upper()}={v:.3f}" for k, v in zip(metric_names, metrics)]
                    )
                    desc = f"✅ Epoch {epoch + 1:{len(str(cfg.epochs))}d}/{cfg.epochs} @ {'DS_Avg':<{max_name_len}}"
                    print(f"{desc}          NUM={num:>4}, {metrics_str}")
                    weighted_metrics = [
                        np.array(score[4:]) * score[3] for score in epoch_scores[1:-1]
                    ]
                    metrics = np.sum(weighted_metrics, 0) / num
                    score = [epoch + 1, lr, "val_Img-Avg", num, *metrics]
                    epoch_scores.append(score)
                    metrics_str = ", ".join(
                        [f"{k.upper()}={v:.3f}" for k, v in zip(metric_names, metrics)]
                    )
                    desc = f"✅ Epoch {epoch + 1:{len(str(cfg.epochs))}d}/{cfg.epochs} @ {'Img_Avg':<{max_name_len}}"
                    print(f"{desc}          NUM={num:>4}, {metrics_str}")
                results.extend(
                    [[fold_idx, *score] for score in epoch_scores]
                    if fold_idx
                    else epoch_scores
                )
                _save_results(cfg, results)
                cfg.weight_dir.mkdir(parents=True, exist_ok=True)
                weight_name = f"{cfg.command}_{cfg.start_time}"
                if fold_idx:
                    weight_name += f"_fold{fold_idx}"
                weight_path = cfg.weight_dir.joinpath(weight_name).with_suffix(".pth")
                if cfg.scheduler_type == SchedulerType.plateau:
                    current_lr = trainer.optimizer.param_groups[0]["lr"]
                    trainer.scheduler.step(metrics[0])
                    if trainer.optimizer.param_groups[0]["lr"] != current_lr:
                        print(
                            f"📉 Learning rate reduced from {current_lr:.2e} to {trainer.optimizer.param_groups[0]['lr']:.2e}"
                        )
                        trainer.model.load_state_dict(
                            torch.load(weight_path, weights_only=True)
                        )
                else:
                    trainer.scheduler.step()
                if best_info is None or metrics[0] > best_info["metrics"][0]:
                    best_info = {
                        "fold_idx": fold_idx,
                        "metrics": metrics,
                        "epoch": epoch + 1,
                    }
                    early_stop_counter = 0
                    torch.save(trainer.model.state_dict(), weight_path)
                    print(f"💾 Saved model weights to: {weight_path}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= 2:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

        elif cfg.scheduler_type != SchedulerType.plateau:
            trainer.scheduler.step()
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
    print(
        f"🏆 Best model (Epoch {best_info['epoch']}): IG={
            best_info['metrics'][0]:.4f}, NSS={best_info['metrics'][1]:.4f}"
    )


@app.command("cv", help="Evaluate model generalization through cross-validation")
@precall_param_merge(setup_config)
def kfold_cross_validate(
    ctx: typer.Context,
    num_folds: Annotated[
        int, typer.Option("--num-folds", "-k", help="Number of cross-validation folds")
    ] = HyperConfigs.num_folds,
):
    cfg: HyperConfigs = ctx.obj["cfg"]
    cfg.num_folds = num_folds
    print("Params: ", cfg)
    split_datasets, _ = prepare_datasets(cfg)
    valid_components = {
        ds_cfg.name: (None, None, GaussFilter(ds_cfg.kernel_size))
        for ds_cfg in cfg.dataset_configs
    }
    results = []
    crossval_splits = [dataset.crossval_split(num_folds) for dataset in split_datasets]
    for fold_idx, (train_datasets, valid_datasets) in enumerate(
        np.array(crossval_splits).transpose(1, 2, 0)
    ):
        print(f"📊 Fold {fold_idx + 1:{len(str(num_folds))}d}/{num_folds}:")
        train_loader = _prepare_data_loader(
            cfg, ConcatDataset([dataset() for dataset in train_datasets])
        )
        for name, dataset in zip(
            [ds_cfg.name for ds_cfg in cfg.dataset_configs], valid_datasets
        ):
            dataset = dataset()
            valid_components[name] = (
                _prepare_data_loader(cfg, dataset, True),
                dataset.build_fixation_maps(),
                valid_components[name][2],
            )
        _run_training_loop(cfg, train_loader, valid_components, results, fold_idx)
        if cfg.debug and fold_idx > 0:
            break


@app.command(name="train", help="Train model with optional validation step")
@precall_param_merge(setup_config)
def train_model(
    ctx: typer.Context,
    train_dataset_configs: Annotated[
        List[DatasetConfig],
        typer.Option(
            "--train-dataset",
            "-t",
            help="Training Datasets (use -t before each name)",
            show_default=False,
            show_choices=False,
            case_sensitive=False,
        ),
    ] = [],
    valid_dataset_configs: Annotated[
        List[DatasetConfig],
        typer.Option(
            "--valid-dataset",
            "-v",
            help="Validation datasets (use -v before each name)",
            show_choices=False,
            show_default="no validation",
            case_sensitive=False,
        ),
    ] = [],
):
    cfg: HyperConfigs = ctx.obj["cfg"]
    cfg.train_dataset_configs = train_dataset_configs
    cfg.valid_dataset_configs = valid_dataset_configs
    print("Params: ", cfg)
    train_datasets, valid_datasets = prepare_datasets(cfg)
    valid_dataset_configs = cfg.dataset_configs + valid_dataset_configs
    valid_components = {
        ds_cfg.name: (None, None, GaussFilter(ds_cfg.kernel_size))
        for ds_cfg in valid_dataset_configs
    }
    valid_results = []
    train_loader = _prepare_data_loader(cfg, ConcatDataset(train_datasets))
    for name, dataset in zip(
        [ds_cfg.name for ds_cfg in valid_dataset_configs], valid_datasets
    ):
        valid_components[name] = (
            _prepare_data_loader(cfg, dataset, True),
            dataset.build_fixation_maps(),
            valid_components[name][2],
        )
    _run_training_loop(cfg, train_loader, valid_components, valid_results)


@app.command("valid", help="validate model on specified datasets")
@precall_param_merge(setup_config)
def validate_models(
    ctx: typer.Context,
    models: Annotated[
        List[ModelConfig],
        typer.Option(
            "--model",
            "-m",
            help="Model to validate (use -m before each name)",
            show_choices=False,
            case_sensitive=False,
        ),
    ],
    valid_dataset_configs: Annotated[
        List[DatasetConfig],
        typer.Option(
            "--valid-dataset",
            "-v",
            help="Validation datasets (use -v before each name)",
            show_choices=False,
            show_default="no validation",
            case_sensitive=False,
        ),
    ] = [],
    gaussian_noise_std: Annotated[
        float,
        typer.Option(
            "--gaussian_noise_std",
            help="Standard deviation of Gaussian noise",
        ),
    ] = 0,
    gaussian_blur_sigma: Annotated[
        float,
        typer.Option(
            "--gaussian_blur_sigma",
            help="Standard deviation of Gaussian noise",
        ),
    ] = 0,
    salt_pepper_density: Annotated[
        float,
        typer.Option(
            "--salt_pepper_density",
            help="Density of salt and pepper noise",
        ),
    ] = 0,
):
    cfg: HyperConfigs = ctx.obj["cfg"]
    print("Params: ", cfg)
    cfg.valid_dataset_configs = valid_dataset_configs
    _, valid_datasets = prepare_datasets(cfg)
    valid_dataset_configs = cfg.dataset_configs + valid_dataset_configs
    valid_components = {
        ds_cfg.name: (None, None, GaussFilter(ds_cfg.kernel_size))
        for ds_cfg in valid_dataset_configs
    }
    for name, dataset in zip(
        [ds_cfg.name for ds_cfg in valid_dataset_configs], valid_datasets
    ):
        if gaussian_noise_std > 0:
            dataset.gaussian_noise_std = gaussian_noise_std
        if gaussian_blur_sigma > 0:
            dataset.gaussian_blur_sigma = gaussian_blur_sigma
        if salt_pepper_density > 0:
            dataset.salt_pepper_density = salt_pepper_density
        valid_components[name] = (
            _prepare_data_loader(cfg, dataset, True),
            dataset.build_fixation_maps(),
            valid_components[name][2],
        )
    max_name_len = np.max([len(name) for name in valid_components.keys()])
    scores = []
    for model in models:
        trainer = _create_trainer(cfg, model.get_model())
        print(f"🔄 Switching to model: {model.name}")
        with torch.no_grad():
            trainer.model.eval()
            model_scores = []
            for name, valid_component in valid_components.items():
                desc = f"✅ {name:<{max_name_len}}"
                metrics = trainer.valid_epoch(valid_component, desc, cfg.quick_valid)
                num = len(valid_component[0])
                score = [model.name, None, f"val_{name}", num, *metrics]
                model_scores.append(score)
            if len(valid_components) > 1:
                metric_names = ["IG", "NSS", "AUC", "SAUC", "CC", "KLD", "SIM"]
                num = sum([score[3] for score in model_scores])
                metrics = np.mean([score[4:] for score in model_scores], 0)
                score = [model.name, None, "val_DS_Avg", num, *metrics]
                model_scores.append(score)
                metrics_str = ", ".join(
                    [f"{k.upper()}={v:.3f}" for k, v in zip(metric_names, metrics)]
                )
                desc = f"✅ {'DS_Avg':<{max_name_len}}"
                print(f"{desc}          NUM={num:>4}, {metrics_str}")
                weighted_metrics = [
                    np.array(score[4:]) * score[3] for score in model_scores[:-1]
                ]
                metrics = np.sum(weighted_metrics, 0) / num
                score = [model.name, None, "val_Img_Avg", num, *metrics]
                model_scores.append(score)
                metrics_str = ", ".join(
                    [f"{k.upper()}={v:.3f}" for k, v in zip(metric_names, metrics)]
                )
                desc = f"✅ {'Img_Avg':<{max_name_len}}"
                print(f"{desc}          NUM={num:>4}, {metrics_str}")
            scores.extend(model_scores)
    _save_results(cfg, scores)


@app.command("draw", help="draw attention maps")
@precall_param_merge(setup_config)
def draw_attention_maps(
    ctx: typer.Context,
    models: Annotated[
        List[ModelConfig],
        typer.Option(
            "--model",
            "-m",
            help="Model to validate (use -m before each name)",
            show_choices=False,
            case_sensitive=False,
        ),
    ],
    valid_dataset_configs: Annotated[
        List[DatasetConfig],
        typer.Option(
            "--valid-dataset",
            "-v",
            help="Validation datasets (use -v before each name)",
            show_choices=False,
            show_default="no validation",
            case_sensitive=False,
        ),
    ] = [],
):
    import matplotlib.pyplot as plt

    cfg: HyperConfigs = ctx.obj["cfg"]
    print("Params: ", cfg)
    cfg.valid_dataset_configs = valid_dataset_configs
    _, valid_datasets = prepare_datasets(cfg)
    valid_dataset_configs = cfg.dataset_configs + valid_dataset_configs
    valid_components = {
        ds_cfg.name: (None, None, GaussFilter(ds_cfg.kernel_size))
        for ds_cfg in valid_dataset_configs
    }
    for name, dataset in zip(
        [ds_cfg.name for ds_cfg in valid_dataset_configs], valid_datasets
    ):
        valid_components[name] = (
            _prepare_data_loader(cfg, dataset, True),
            dataset.build_fixation_maps(),
            valid_components[name][2],
        )
    for model in models:
        trainer = _create_trainer(cfg, model.get_model())
        print(f"🔄 Switching to model: {model.name}")
        with torch.no_grad():
            trainer.model.eval()
            for name, valid_component in valid_components.items():
                for i, (images, shapes, coords) in enumerate(valid_component[0]):
                    if i < 42:
                        continue
                    plt.imshow(images.cpu()[0].permute(1, 2, 0))
                    plt.axis("off")
                    plt.savefig(
                        f"Documents/{name}_input.png",
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    if trainer.model._get_name() == "DeepGazeIIE":
                        centerbias = torch.tensor(
                            np.load(Path(DEEPGAZE_PATH) / "centerbias_mit1003.npy")
                        ).to(trainer.device)[None, None, :, :]
                        centerbias = torch.nn.functional.interpolate(
                            centerbias, size=shapes[0][:2], mode="nearest"
                        )
                        centerbias -= torch.logsumexp(centerbias, dim=(2, 3))
                        output = trainer.model(
                            images.to(trainer.device) * 255,
                            centerbias=centerbias.squeeze(0),
                        ).squeeze(1)
                    elif trainer.model._get_name() == "UNISAL":
                        ar = shapes[0][0] / shapes[0][1]
                        min_prod = 100
                        max_prod = 120
                        ar_array = []
                        size_array = []
                        for n1 in range(7, 14):
                            for n2 in range(7, 14):
                                if min_prod <= n1 * n2 <= max_prod:
                                    this_ar = n1 / n2
                                    ar_array.append(
                                        min((ar, this_ar)) / max((ar, this_ar))
                                    )
                                    size_array.append((n1, n2))
                        bn_size = size_array[np.argmax(np.array(ar_array)).item()]
                        out_size = tuple(r * 32 for r in bn_size)
                        images = torch.nn.functional.interpolate(
                            images, size=out_size, mode="bilinear", align_corners=False
                        )
                        images = trainer.normalizer(images)
                        output = trainer.model(
                            images.to(trainer.device).unsqueeze(0), static=True
                        )
                        output = torch.nn.functional.interpolate(
                            output.squeeze(1),
                            size=shapes[0][:2],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    else:
                        output = trainer.model(images.to(trainer.device)).squeeze(1)
                    output = output.exp()
                    output = output / output.sum()
                    current_fixation_maps = [
                        torch.sparse_coo_tensor(
                            coord,
                            torch.ones(len(coord[0]), dtype=torch.long),
                            shape[:2],
                        ).coalesce()
                        for shape, coord in zip(shapes, coords)
                    ]
                    gauss_map = valid_component[2](
                        torch.stack(current_fixation_maps)
                        .to(cfg.device)
                        .to_dense()
                        .unsqueeze(1)
                        .float()
                    ).squeeze(1)
                    gauss_map /= gauss_map.sum()
                    vmin = gauss_map.min()
                    vmax = gauss_map.max()
                    plt.imshow(output.cpu()[0], vmin=vmin, vmax=vmax, cmap="jet")
                    plt.axis("off")
                    plt.savefig(
                        f"Documents/{name}_output_{model.name}.png",
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.imshow(gauss_map.cpu()[0], vmin=vmin, vmax=vmax, cmap="jet")
                    plt.axis("off")
                    plt.savefig(
                        f"Documents/{name}_gauss.png",
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    break
                pass


if __name__ == "__main__":
    app()
