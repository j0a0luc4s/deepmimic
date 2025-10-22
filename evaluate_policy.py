from deepmimic.env import T1Env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


env = T1Env(render_mode="human")
eval_env = Monitor(T1Env(render_mode="human"))

model = PPO(
    "MlpPolicy",
    env,
    device="cpu",
)

model.learn(
    1e5,
)

env.close()

evaluate_policy(
    model,
    eval_env,
    render=True,
)

eval_env.close()
