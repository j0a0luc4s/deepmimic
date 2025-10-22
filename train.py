from typing import Callable

import time

from deepmimic.env import T1Env

from gymnasium import spaces

import torch

from stable_baselines3 import PPO
from stable_baselines3.common import policies, callbacks, vec_env, env_util


class SeparateLRPolicy(policies.ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        actor_params = (
            list(self.action_net.parameters()) +
            list(self.mlp_extractor.policy_net.parameters())
        )
        critic_params = (
            list(self.value_net.parameters()) +
            list(self.mlp_extractor.value_net.parameters())
        )

        optimizer = torch.optim.Adam([
            {'params': actor_params,  'lr': self.actor_learning_rate},
            {'params': critic_params, 'lr': self.critic_learning_rate},
        ])

        return optimizer


env = env_util.make_vec_env(
    env_id=T1Env(ref="t1_mjp_joystick", episode_length=1000),
    n_envs=16,
    vec_env_cls=vec_env.SubprocVecEnv,
)

eval_env = T1Env(
    ref="t1_mjp_joystick",
    episode_length=1000,
    render_mode="human"
)

model = PPO(
    policy=SeparateLRPolicy,
    env=env,
    n_steps=1024,
    batch_size=256,
    n_epochs=1,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    rollout_buffer_class=None,
    rollout_buffer_kwargs=None,
    target_kl=None,
    stats_window_size=100,
    tensorboard_log="tensorboard_log",
    policy_kwargs=dict(actor_learning_rate=5e-5, critic_learning_rate=1e-2),
    verbose=1,
    seed=None,
    device="cpu",
    _init_setup_model=True,
)

model.learn(
    total_timesteps=50_000_000,
    callback=callbacks.CallbackList([
        callbacks.EvalCallback(
            eval_env=eval_env,
            eval_freq=100_000,
        ),
        callbacks.CheckpointCallback(
            save_freq=100_000,
            save_path="saves",
            name_prefix=f"rl_model_{int(time.time())}",
        ),
    ]),
    progress_bar=True,
)
