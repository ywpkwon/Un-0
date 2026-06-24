# ruff: noqa: INP001
"""ImageNet-64 preprocessing for image generation training/inference.

`preprocess_imagenet` is the single entry point: it maps a source PIL image to
the `(64, 64, 3)` uint8 array used for training. This matches the preprocessing
behind OpenAI's `VIRTUAL_imagenet64_labeled.npz` reference batch and the ADM
training pipeline in `openai/guided-diffusion`
(`guided_diffusion/image_datasets.py`). The pixels therefore line up with that
reference, so running OpenAI's ADM evaluator against `VIRTUAL_imagenet64_labeled.npz`
yields FID/sFID/IS/Precision/Recall directly comparable to ADM, DiT, EDM, and
EDM2. That evaluator is not included here; the in-training FID this repo computes
(`un0/metrics.py`) is clean-FID against custom validation statistics, which is a
different number — a fast training-time proxy, not the headline ADM-evaluator FID.

The pre-built ImageNet-64 dataset was produced by running this on each image
and storing the result losslessly. If you persist these arrays, use a lossless
format (e.g. PNG); a lossy format such as JPEG would shift the pixels and move
FID.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 64


def preprocess_imagenet(image: Image.Image) -> np.ndarray:
    """ImageNet training/inference preprocessing."""
    # Apply the source image's EXIF orientation before any resampling. The
    # reference dataset was built by loading images through HuggingFace
    # `load_dataset`, which bakes in orientation, so this is a no-op on that
    # path; it keeps the result correct when a caller loads the image with a
    # raw `Image.open`, which would otherwise mis-crop the ~0.02% of ImageNet
    # images carrying an EXIF rotation tag.
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    target = IMAGE_SIZE

    while min(image.size) >= 2 * target:
        image = image.resize(
            (image.size[0] // 2, image.size[1] // 2),
            resample=Image.BOX,
        )

    scale = target / min(image.size)
    new_size = (
        round(image.size[0] * scale),
        round(image.size[1] * scale),
    )
    image = image.resize(new_size, resample=Image.BICUBIC)

    arr = np.array(image.convert("RGB"))
    crop_y = (arr.shape[0] - target) // 2
    crop_x = (arr.shape[1] - target) // 2
    return arr[crop_y : crop_y + target, crop_x : crop_x + target]
