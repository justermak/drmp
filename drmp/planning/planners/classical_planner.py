from abc import ABC, abstractmethod
from typing import Any, Dict, List

from drmp.universe.environments import EnvBase
from drmp.universe.robot import RobotBase
import torch


class ClassicalPlanner(ABC):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
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
    def optimize(
        self, **kwargs
    ) -> torch.Tensor | List[torch.Tensor]:
        pass
