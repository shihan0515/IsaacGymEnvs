from typing import Optional

import torch

from isaacgymenvs.utils.torch_jit_utils import quat_conjugate, quat_mul


def ik(
    jacobian_end_effector: torch.Tensor,
    current_position: torch.Tensor,
    current_orientation: torch.Tensor,
    goal_position: torch.Tensor,
    goal_orientation: Optional[torch.Tensor] = None,
    damping_factor: float = 0.05,
    squeeze_output: bool = True,
) -> torch.Tensor:
    """
    Inverse kinematics using damped least squares method

    :param jacobian_end_effector: End effector's jacobian
    :type jacobian_end_effector: torch.Tensor
    :param current_position: End effector's current position
    :type current_position: torch.Tensor
    :param current_orientation: End effector's current orientation
    :type current_orientation: torch.Tensor
    :param goal_position: End effector's goal position
    :type goal_position: torch.Tensor
    :param goal_orientation: End effector's goal orientation (default: None)
    :type goal_orientation: torch.Tensor or None
    :param damping_factor: Damping factor (default: 0.05)
    :type damping_factor: float
    :param squeeze_output: Squeeze output (default: True)
    :type squeeze_output: bool

    :return: Change in joint angles
    :rtype: torch.Tensor
    """
    if goal_orientation is None:
        goal_orientation = current_orientation

    # compute error
    q = quat_mul(goal_orientation, quat_conjugate(current_orientation))
    error = torch.cat(
        [
            goal_position - current_position,  # position error
            q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1),
        ],  # orientation error
        dim=-1,
    ).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    _lambda = torch.eye(6, device=jacobian_end_effector.device) * (
        damping_factor**2
    )
    if squeeze_output:
        return (
            transpose
            @ torch.inverse(jacobian_end_effector @ transpose + _lambda)
            @ error
        ).squeeze(dim=2)
    else:
        return (
            transpose
            @ torch.inverse(jacobian_end_effector @ transpose + _lambda)
            @ error
        )