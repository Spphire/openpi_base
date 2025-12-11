import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bimanual_iphone_flexiv_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/left_wrist_img": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_img": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "actions": np.random.rand(14),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BimanualiPhoneFlexivInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        left_wrist_image = _parse_image(data["observation/left_wrist_img"])
        right_wrist_image = _parse_image(data["observation/right_wrist_img"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist_image),
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "prev_state" in data:
            inputs["prev_state"] = data["prev_state"]

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualiPhoneFlexivOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}
