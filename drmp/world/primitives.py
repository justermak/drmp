from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from drmp.config import DEFAULT_TENSOR_ARGS
from drmp.utils.torch_utils import to_torch


class PrimitiveShapeField(ABC):
    def __init__(self, tensor_args: Dict[str, Any] = DEFAULT_TENSOR_ARGS) -> None:
        self.tensor_args = tensor_args

    @abstractmethod
    def compute_signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    

class MultiSphereField(PrimitiveShapeField): 
    def __init__(self, centers: torch.Tensor, radii: torch.Tensor, tensor_args: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.radii = to_torch(radii, **self.tensor_args)

    def compute_signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        distance_to_centers = torch.cdist(x, self.centers)
        sdfs = distance_to_centers - self.radii
        return sdfs


class MultiBoxField(PrimitiveShapeField):
    def __init__(
        self, 
        centers: torch.Tensor, 
        half_sizes: torch.Tensor, 
        smooth_factor: float = 0.3, 
        tensor_args: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.half_sizes = to_torch(half_sizes, **self.tensor_args)
        self.smooth_factor = smooth_factor
        self.radii = torch.min(self.half_sizes, dim=-1).values * self.smooth_factor

    def compute_signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        distance_to_centers = torch.abs(x.unsqueeze(-2) - self.centers)
        q = distance_to_centers - self.half_sizes + self.radii.unsqueeze(-1)
        max_q = torch.amax(q, dim=-1)
        sdfs = (
            torch.minimum(max_q, torch.zeros_like(max_q))
            + torch.linalg.norm(torch.relu(q), dim=-1)
            - self.radii
        )
        return sdfs


class ObjectField(PrimitiveShapeField):
    fields: List[PrimitiveShapeField]
    
    def __init__(self, primitive_fields: List[PrimitiveShapeField]) -> None:
        assert primitive_fields is not None and isinstance(primitive_fields, List)
        assert len(primitive_fields) > 0, "ObjectField must contain at least one primitive"
        
        super().__init__(tensor_args=primitive_fields[0].tensor_args)
        self.fields = primitive_fields

    def compute_signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        sdf_fields = [field.compute_signed_distance(x) for field in self.fields]
        return torch.concat(sdf_fields, dim=-1)
    