# Path Configuration - MODIFY THESE BEFORE RUNNING
# ------------------------------------------------
DATASET_DIR = "data"
# Dataset location (data will download here automatically)

PYSALIENCY_PATH = "libs/pysaliency"
# Required for dataset loading. Git clone here, or comment if using pip-installed pysaliency

DEEPGAZE_PATH = "libs/DeepGaze"
# For DeepGaze comparison. Git clone here, or comment if not needed

UNISAL_PATH = "libs/unisal"
# For UNISAL comparison. Git clone here, or comment if not needed
# ------------------------------------------------

# ruff: noqa: E402
import sys
from enum import Enum
from pathlib import Path
import torch

sys.path.extend([PYSALIENCY_PATH, DEEPGAZE_PATH, UNISAL_PATH])
from pysaliency.external_datasets import (
    get_cat2000_train,
    get_COCO_Freeview_train,
    get_COCO_Freeview_validation,
    get_COCO_Search18_train,
    get_COCO_Search18_validation,
    get_iSUN_training,
    get_iSUN_validation,
    get_mit1003,
    get_SALICON_train,
    get_SALICON_val,
    get_toronto,
)


class SchedulerType(Enum):
    cosine = "cosine"
    plateau = "plateau"
    exponential = "exponential"


class ModelConfig(Enum):
    SALMOVIT = "salmovit"
    DEEPGAZE_IIE = "deepgaze_iie"
    UNISAL = "unisal"

    def get_model(self):
        if self == ModelConfig.SALMOVIT:
            from salmovit_model import SalMoViT

            model = SalMoViT()
            model.load_state_dict(
                torch.load(
                    Path(__file__).parent / "weights_best.pth", weights_only=False
                )
            )
            return model

        elif self == ModelConfig.DEEPGAZE_IIE:
            import deepgaze_pytorch

            return deepgaze_pytorch.DeepGazeIIE(pretrained=True)

        elif self == ModelConfig.UNISAL:
            from unisal.model import UNISAL as UnisalModel

            model = UnisalModel()
            model.load_state_dict(
                torch.load(
                    Path(UNISAL_PATH)
                    / "training_runs/pretrained_unisal/weights_best.pth",
                    weights_only=False,
                )
            )
            return model


class DatasetConfig(Enum):
    MIT1003 = (lambda: get_mit1003(location=DATASET_DIR), 24)
    CAT2000 = (lambda: get_cat2000_train(location=DATASET_DIR), 32)
    CAT2000_V1_1 = (lambda: get_cat2000_train(location=DATASET_DIR, version="1.1"), 32)
    COCO_FREEVIEW = (lambda: get_COCO_Freeview_train(location=DATASET_DIR), 40)
    COCO_FREEVIEW_VALIDATION = (
        lambda: get_COCO_Freeview_validation(location=DATASET_DIR),
        40,
    )
    COCO_SEARCH18 = (lambda: get_COCO_Search18_train(location=DATASET_DIR), 40)
    COCO_SEARCH18_VALIDATION = (
        lambda: get_COCO_Search18_validation(location=DATASET_DIR),
        40,
    )
    SALICON_TRAIN_2017_FIXATIONS = (
        lambda: get_SALICON_train(
            location=DATASET_DIR, edition="2017", fixation_type="fixations"
        ),
        24,
    )
    SALICON_TRAIN_2017_MOUSE = (
        lambda: get_SALICON_train(
            location=DATASET_DIR, edition="2017", fixation_type="mouse"
        ),
        24,
    )
    SALICON_VALID_2017_FIXATIONS = (
        lambda: get_SALICON_val(
            location=DATASET_DIR, edition="2017", fixation_type="fixations"
        ),
        24,
    )
    SALICON_VALID_2017_MOUSE = (
        lambda: get_SALICON_val(
            location=DATASET_DIR, edition="2017", fixation_type="mouse"
        ),
        24,
    )
    TORONTO = (lambda: get_toronto(location=DATASET_DIR), 24)
    iSUN = (lambda: get_iSUN_training(location=DATASET_DIR), 24)
    iSUN_VALIDATION = (lambda: get_iSUN_validation(location=DATASET_DIR), 24)

    def __init__(self, loader, kernel_size):
        self.loader = loader
        self.kernel_size = kernel_size
        Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)

    @property
    def value(self):
        return self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance)
