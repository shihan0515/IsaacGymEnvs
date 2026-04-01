import isaacgym
import isaacgymenvs
import torch

num_envs = 4

envs = isaacgymenvs.make(
    seed=0,
    task="DiabloGraspCustom3",
    num_envs=num_envs,
    rl_device="cuda",
    sim_device="cuda",
    graphics_device_id=0,
)

obs = envs.reset()
for i in range(10000000):
    zero_actions = envs.zero_actions()
    envs.step(zero_actions)