import math
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms.functional import gaussian_blur


class SaliencyDataset(Dataset):
    def __init__(
        self,
        stimuli,
        fixations,
        transforms: Compose | str = "default",
        indices: np.ndarray = None,
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.transforms = (
            Compose(
                [
                    ToTensor(),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )
            if transforms == "default"
            else transforms
        )
        self.indices = indices if indices is not None else np.unique(fixations.n)
        self.gaussian_blur_sigma = 0.0
        self.gaussian_noise_std = 0.0
        self.salt_pepper_density = 0.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            img = self.stimuli.stimuli[self.indices[idx]]
            shape = self.stimuli.shapes[self.indices[idx]]
            if self.gaussian_blur_sigma > 0:
                img = gaussian_filter(img, self.gaussian_blur_sigma)
            if self.transforms:
                img = self.transforms(img)
            if self.gaussian_noise_std > 0:
                img += torch.randn_like(img) * self.gaussian_noise_std
                img = img.clamp(0, 1)
            if self.salt_pepper_density > 0:
                mask = torch.rand(img.shape[1], img.shape[2])
                img[:, mask < (self.salt_pepper_density / 2)] = 0
                img[:, mask > (1 - self.salt_pepper_density / 2)] = 1
            fix_selected = self.fixations.n == self.indices[idx]
            coords = np.stack(
                [
                    np.clip(self.fixations.y_int[fix_selected], 0, shape[0] - 1),
                    np.clip(self.fixations.x_int[fix_selected], 0, shape[1] - 1),
                ]
            )
            return (img, shape, coords)
        return SaliencyDataset(
            self.stimuli, self.fixations, self.transforms, self.indices[idx]
        )

    def build_fixation_maps(self):
        fix_mask = np.isin(self.fixations.n, self.indices)
        shapes = np.array([s[:2] for s in self.stimuli.shapes])
        unique_shapes = np.unique(shapes[self.indices], axis=0)
        coords = np.column_stack(
            (self.fixations.y[fix_mask], self.fixations.x[fix_mask])
        )
        scaled_coords = np.clip(
            (
                coords / shapes[self.fixations.n[fix_mask]] * unique_shapes[:, None, :]
            ).astype(int),
            0,
            unique_shapes[:, None, :] - 1,
        )
        return {
            tuple(unique_shape): torch.sparse_coo_tensor(
                scaled_coords[i].T,
                torch.ones(fix_mask.sum(), dtype=torch.int),
                tuple(unique_shape),
            ).coalesce()
            for i, unique_shape in enumerate(unique_shapes)
        }

    def split(self, test_size=0.2, random_state=None):
        train_indices, valid_indices = train_test_split(
            range(len(self)), test_size=test_size, random_state=random_state
        )
        return (self[train_indices], self[valid_indices])

    def crossval_split(self, n_splits=10):
        kf = KFold(n_splits, shuffle=True)
        return [
            (lambda: self[train_indices], lambda: self[valid_indices])
            for train_indices, valid_indices in kf.split(self)
        ]


def split_component_collate(batch):
    return (
        default_collate([x[0] for x in batch]),
        *zip(*[x[1:] for x in batch]),
    )
