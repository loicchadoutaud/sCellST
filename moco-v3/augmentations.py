import random

from torch import Tensor
from torchvision import transforms
from torchvision.transforms.v2 import Transform
from PIL import ImageFilter, ImageOps


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
