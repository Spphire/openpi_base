from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

p = "/mnt/model/.cache/huggingface/hub/lerobot/flexiv/place1102"
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(p)
dataset = lerobot_dataset.LeRobotDataset(
    'flexiv/place1102',
    root=p,
    delta_timestamps={
        key: [t / dataset_meta.fps for t in range(1)] for key in ['actions']
    },
)
print('ok')