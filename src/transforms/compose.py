from typing import Callable 
import torch

class Compose:
    """
    Apply a list of transforms sequentially to a tensor.
    """

    def __init__(self,
                 transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:

        self.transforms = transforms

    def __call__(self,
                 x: torch.Tensor) -> torch.Tensor:
        """
        Apply the transforms to the input tensor.
        """
        for transform in self.transforms:
            x = transform(x)
        return x