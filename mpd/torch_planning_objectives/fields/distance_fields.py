from abc import ABC, abstractmethod

import einops
import torch

import torch.nn.functional as Functional


class DistanceField(ABC):
    def __init__(self, tensor_args=None):
        self.tensor_args = tensor_args

    def compute_cost(self, q, link_pos, *args, **kwargs):
        link_orig_shape = link_pos.shape
        if len(link_orig_shape) == 2:
            h = 1
            b, d = link_orig_shape
            link_pos = einops.rearrange(
                link_pos, "b d -> b 1 d"
            )  # add dimension of task space link
        elif len(link_orig_shape) == 3:
            h = 1
            b, t, d = link_orig_shape
        elif (
            len(link_orig_shape) == 4
        ):  # batch, horizon, num_links, 3  # position tensor
            b, h, t, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d -> (b h) t d")
        elif (
            len(link_orig_shape) == 5
        ):  # batch, horizon, num_links, 4, 4  # homogeneous transform tensor
            b, h, t, d, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d d -> (b h) t d d")
        else:
            raise NotImplementedError

        cost = self.compute_costs_impl(q, link_pos, *args, **kwargs)

        if cost.ndim == 1:
            cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        return cost

    @abstractmethod
    def compute_costs_impl(self, *args, **kwargs):
        pass


def interpolate_points(points, num_interpolated_points):
    # https://github.com/SamsungLabs/RAMP/blob/c3bd23b2c296c94cdd80d6575390fd96c4f83d83/mppi_planning/cost/collision_cost.py#L89
    points = Functional.interpolate(
        points.transpose(-2, -1),
        size=num_interpolated_points,
        mode="linear",
        align_corners=True,
    ).transpose(-2, -1)
    return points


class EmbodimentDistanceFieldBase(DistanceField):
    def __init__(
        self,
        robot,
        link_idxs_for_collision_checking=None,
        num_interpolated_points=30,
        collision_margins=0.0,
        cutoff_margin=0.001,
        field_type="sdf",
        clamp_sdf=True,
        interpolate_link_pos=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert robot is not None, (
            "You need to pass a robot instance to the embodiment distance fields"
        )
        self.robot = robot
        self.link_idxs_for_collision_checking = link_idxs_for_collision_checking
        self.num_interpolated_points = num_interpolated_points
        self.collision_margins = collision_margins
        self.cutoff_margin = cutoff_margin
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.interpolate_link_pos = interpolate_link_pos

    def compute_embodiment_cost(
        self, q, link_pos, field_type=None, **kwargs
    ):  # position tensor
        if field_type is None:
            field_type = self.field_type
        if field_type == "sdf":
            # this computes the negative cost from the DISTANCE FUNCTION
            margin = self.collision_margins + self.cutoff_margin
            # returns all distances from each link to the environment
            margin_minus_sdf = -(
                self.compute_embodiment_signed_distances(q, link_pos, **kwargs) - margin
            )
            if self.clamp_sdf:
                clamped_sdf = torch.relu(margin_minus_sdf)
            else:
                clamped_sdf = margin_minus_sdf
            if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
                clamped_sdf = clamped_sdf.max(-2)[0]
            # sum over link points for gradient computation
            return clamped_sdf.sum(-1)
        elif field_type == "occupancy":
            return self.compute_embodiment_collision(q, link_pos, **kwargs)
            # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
            # return (distances < margin).sum(-1)
        else:
            raise NotImplementedError(
                "field_type {} not implemented".format(field_type)
            )

    def compute_costs_impl(self, q, link_pos, **kwargs):
        # position link_pos tensor # batch x num_links x 3
        # interpolate to approximate link spheres
        link_pos_robot = link_pos

        link_pos_robot = link_pos_robot[..., self.link_idxs_for_collision_checking, :]
        if self.interpolate_link_pos:
            # select the robot links used for collision checking
            link_pos = interpolate_points(link_pos_robot, self.num_interpolated_points)
        else:
            link_pos = link_pos_robot

        embodiment_cost = self.compute_embodiment_cost(q, link_pos, **kwargs)
        return embodiment_cost

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_collision(self, *args, **kwargs):
        raise NotImplementedError


class CollisionObjectBase(EmbodimentDistanceFieldBase):
    def __init__(
        self, *args, link_margins_for_object_collision_checking_tensor=None, **kwargs
    ):
        super().__init__(
            *args,
            collision_margins=link_margins_for_object_collision_checking_tensor,
            **kwargs,
        )

    def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):
        return self.object_signed_distances(link_pos, **kwargs)

    def compute_embodiment_collision(self, q, link_pos, **kwargs):
        # position tensor
        margin = kwargs.get("margin", self.collision_margins + self.cutoff_margin)
        signed_distances = self.object_signed_distances(link_pos, **kwargs)
        collisions = signed_distances < margin
        # reduce over points (dim -1) and over objects (dim -2)
        any_collision = torch.any(torch.any(collisions, dim=-1), dim=-1)
        return any_collision

    @abstractmethod
    def object_signed_distances(self, *args, **kwargs):
        raise NotImplementedError


class CollisionObjectDistanceField(CollisionObjectBase):
    def __init__(self, *args, df_obj_list_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_obj_list_fn = df_obj_list_fn

    def object_signed_distances(self, link_pos, **kwargs):
        if self.df_obj_list_fn is None:
            return torch.inf
        df_obj_list = self.df_obj_list_fn()
        link_dim = link_pos.shape[:-1]
        link_pos = link_pos.reshape(
            -1, link_pos.shape[-1]
        )  # flatten batch_dim and links
        dfs = []
        for df in df_obj_list:
            dfs.append(
                df.compute_signed_distance(link_pos).view(link_dim)
            )  # df() returns batch_dim x links
        return torch.stack(dfs, dim=-2)  # batch_dim x num_sdfs x links


class CollisionWorkspaceBoundariesDistanceField(CollisionObjectBase):
    def __init__(self, *args, ws_bounds_min=None, ws_bounds_max=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_min = ws_bounds_min
        self.ws_max = ws_bounds_max

    def object_signed_distances(self, link_pos, **kwargs):
        signed_distances_bounds_min = link_pos - self.ws_min
        signed_distances_bounds_min = torch.sign(
            signed_distances_bounds_min
        ) * torch.abs(signed_distances_bounds_min)
        signed_distances_bounds_max = self.ws_max - link_pos
        signed_distances_bounds_max = torch.sign(
            signed_distances_bounds_max
        ) * torch.abs(signed_distances_bounds_max)
        signed_distances_bounds = torch.cat(
            (signed_distances_bounds_min, signed_distances_bounds_max), dim=-1
        )
        return signed_distances_bounds.transpose(-2, -1)  # batch_dim x num_sdfs x links
