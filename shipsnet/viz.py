from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
from jsonargparse.typing import PositiveInt


def _prepare_array(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().numpy()

    assert array.ndim == 3, "Expected array to have 3 dimensions"
    assert array.shape[0] == 3, "Expected first dimension to be RGB channels"

    if array.max() <= 1:
        array *= 255  # rescale to 8 bit pixels

    # 8-bit
    array = array.astype("uint8")

    # Image should have dims (row, col, channel)
    array = array.transpose([1, 2, 0])

    return array


def array_to_rgb_image(array: Union[np.ndarray, torch.Tensor]) -> Image:
    array = _prepare_array(array)
    image = Image.fromarray(array, "RGB")
    return image


def array_to_rgb_histogram(
    array: Union[np.ndarray, torch.Tensor],
    bins: PositiveInt = 30,
    embed_image: bool = False,
) -> plt.Figure:
    array = _prepare_array(array)
    r, g, b = [channel.flatten() for channel in array]

    fig, ax = plt.subplots()
    ax.set_xlabel("pixel intensity")
    ax.set_ylabel("count")
    ax.set_xlim(0, 255)
    ax.hist(r, bins=30, color="r", alpha=0.5)
    ax.hist(g, bins=30, color="g", alpha=0.5)
    ax.hist(b, bins=30, color="b", alpha=0.5)

    # Embed image in the top-right (high count + intensity) corner
    if embed_image:
        im = fig.add_axes([0.6, 0.54, 0.33, 0.33])
        im.set_axis_off()
        im.imshow(array)

    return fig
