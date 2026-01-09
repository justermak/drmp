from drmp.world.environments import EnvBase
from drmp.world.robot import Robot
import torch

from typing import Dict, Any, List
from abc import ABC, abstractmethod


class ClassicalPlanner(ABC):
    def __init__(
        self,
        env: EnvBase,
        robot: Robot,
        use_extra_objects: bool,
        tensor_args: Dict[str, Any],
    ):
        self.env = env
        self.robot = robot
        self.use_extra_objects = use_extra_objects
        self.tensor_args = tensor_args
        self.start_pos: torch.Tensor = None
        self.goal_pos: torch.Tensor = None
        
    @abstractmethod
    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        pass
    
    @abstractmethod
    def optimize(self, n_trajectories: int, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        pass    
    