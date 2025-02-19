import random

import numpy as np
from scipy import linalg
import skimage.color as skimage_color
import torch
from PIL import ImageFilter, ImageOps, Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.v2 import Transform


# Hematoxylin + Eosin
RGB_FROM_HEX = np.array([[0.650, 0.704, 0.286], [0.07, 0.99, 0.11], [0.0, 0.0, 0.0]])
RGB_FROM_HEX[2, :] = np.cross(RGB_FROM_HEX[0, :], RGB_FROM_HEX[1, :])
HEX_FROM_RGB = linalg.inv(RGB_FROM_HEX)

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class Equalize(object):
    def __call__(self, x):
        return ImageOps.equalize(x)


class Posterize(object):
    def __call__(self, x):
        return ImageOps.posterize(x, 4)


class RotationCrop(Transform):
    def __init__(self, degrees: int, size: int):
        super().__init__()
        self.degrees = degrees
        self.size = size
        augmentation = [
            transforms.RandomApply([transforms.RandomRotation(self.degrees)], p=1),
            transforms.CenterCrop(size=self.size),
        ]
        self.transform = transforms.Compose(augmentation)

    def __call__(self, x: Tensor) -> Tensor:
        return self.transform(x)


class HEStainAugmentationPil:
    """Deconvolve input tile into H, E, and Residual channels. Then sample one gaussian-random factor for H and for E,
    and scale these stains accordingly. Reapply scaled stains to produced stain-augmented tiles.
    """

    def __init__(
            self,
            gaussian_mean: float | None = None,
            gaussian_std: float | None = None,
            uniform_min: float | None = None,
            uniform_max: float | None = None,
    ):
        assert (gaussian_mean is None and gaussian_std is None) != (
                uniform_min is None and uniform_max is None
        )
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max

    def get_factors(self) -> tuple[float, float] | tuple[Tensor, Tensor]:
        if self.uniform_min is not None and self.uniform_max is not None:
            h_factor = random.uniform(self.uniform_min, self.uniform_max)
            e_factor = random.uniform(self.uniform_min, self.uniform_max)
        elif self.gaussian_mean is not None and self.gaussian_std is not None:
            h_factor = random.gauss(self.gaussian_mean, self.gaussian_std)
            e_factor = random.gauss(self.gaussian_mean, self.gaussian_std)
        else:
            raise ValueError("Should not happen.")
        return h_factor, e_factor

    def __call__(
            self,
            image: Image,
            h_factor: float | None = None,
            e_factor: float | None = None,
    ) -> Image:
        if h_factor is None or e_factor is None:
            h_factor, e_factor = self.get_factors()
        augmented_hex_from_rgb = HEX_FROM_RGB * [[h_factor], [e_factor], [1]]
        separated_augmented_image = skimage_color.separate_stains(
            image, augmented_hex_from_rgb
        )
        augmented_image = skimage_color.combine_stains(
            separated_augmented_image, RGB_FROM_HEX
        )
        return augmented_image

    def __repr__(self) -> str:
        format_string = self.__name__ + "(gaussian_mean={}, gaussian_std={}".format(
            self.gaussian_mean, self.gaussian_std
        )
        return format_string


class HEStainAugmentationTorch:
    """Deconvolve input tile into H, E, and Residual channels. Then sample one gaussian-random factor for H and for E,
    and scale these stains accordingly. Reapply scaled stains to produced stain-augmented tiles.
    """

    def __init__(
            self,
            gaussian_mean: float | None = None,
            gaussian_std: float | None = None,
            uniform_min: float | None = None,
            uniform_max: float | None = None,
            n_updates: int = 1,
    ):
        assert (gaussian_mean is None and gaussian_std is None) != (
                uniform_min is None and uniform_max is None
        )
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max
        self.n_updates = n_updates
        self.rgb_from_hex = torch.tensor(RGB_FROM_HEX, dtype=torch.float32)

    def get_factors(self) -> tuple[float, float] | tuple[Tensor, Tensor]:
        if self.uniform_min is not None and self.uniform_max is not None:
            h_factor = random.uniform(self.uniform_min, self.uniform_max)
            e_factor = random.uniform(self.uniform_min, self.uniform_max)
        elif self.gaussian_mean is not None and self.gaussian_std is not None:
            h_factor = random.gauss(self.gaussian_mean, self.gaussian_std)
            e_factor = random.gauss(self.gaussian_mean, self.gaussian_std)
        else:
            raise ValueError("Should not happen.")
        return h_factor, e_factor

    def _apply(
            self,
            image: Tensor,
            h_factor: float | None = None,
            e_factor: float | None = None,
    ) -> Tensor:
        if h_factor is None or e_factor is None:
            h_factor, e_factor = self.get_factors()
        augmented_hex_from_rgb = torch.tensor(
            HEX_FROM_RGB * [[h_factor], [e_factor], [1]], dtype=torch.float32
        )
        separated_augmented_image = separate_stains_torch(image, augmented_hex_from_rgb)
        augmented_image = combine_stains_torch(
            separated_augmented_image, self.rgb_from_hex
        )
        return augmented_image

    def __call__(
            self,
            image: Tensor,
            h_factor: float | None = None,
            e_factor: float | None = None,
    ) -> Tensor:
        for _ in range(self.n_updates):
            image = self._apply(image, h_factor, e_factor)
        return image

    def __repr__(self) -> str:
        format_string = "H&EStainAugmentation(gaussian_mean={}, gaussian_std={}".format(
            self.gaussian_mean, self.gaussian_std
        )
        return format_string


def separate_stains_torch(
        image: torch.Tensor, stain_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Separates stains in an image in the format (C, W, H) using the given stain matrix.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, W, H).
        stain_matrix (torch.Tensor): Stain matrix of shape (3, 3).

    Returns:
        torch.Tensor: Separated stains tensor of shape (W, H, C).
    """
    # Convert image to (W, H, C) for processing
    image = image.permute(1, 2, 0)  # (C, W, H) -> (W, H, C)

    # Perform log transformation and separation
    log_adjust = torch.log(torch.tensor(1e-6, dtype=torch.float32))
    image = torch.clamp(image, min=1e-6)  # Ensure no values below 1e-6
    log_od = torch.log(image) / log_adjust
    stains = torch.matmul(log_od, stain_matrix)

    # Ensure no negative values
    stains = torch.clamp(stains, min=0)

    return stains


def combine_stains_torch(
        stains: torch.Tensor, stain_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Combines separated stains into an image in the format (W, H, C) using the given stain matrix.

    Args:
        stains (torch.Tensor): Separated stains tensor of shape (W, H, C).
        stain_matrix (torch.Tensor): Stain matrix of shape (3, 3).

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (C, W, H).
    """

    # Perform exponentiation and combination
    log_adjust = -torch.log(torch.tensor(1e-6, dtype=torch.float32))
    log_rgb = -(stains * log_adjust)
    rgb = torch.exp(torch.matmul(log_rgb, stain_matrix))

    # Clip values to [0, 1]
    rgb = torch.clamp(rgb, 0, 1)

    # Convert back to (C, W, H)
    return rgb.permute(2, 0, 1)
