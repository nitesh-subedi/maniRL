from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.ppo import CnnPolicy as ppocnn
from stable_baselines3.sac import CnnPolicy as saccnn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import torch as th
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed

# seed_number = 42
# set_random_seed(seed_number)

name = f"SAC_intrinsic_reward_v3"
run = wandb.init(
    project="plant_manipulation",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
    id=name,  # optional,
)


CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Set display options to show default grid
    "anti_aliasing": 0,
}

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = f"./results/{name}"

# set headless to false to visualize training
my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(vf=[128, 128, 128], pi=[128, 128, 128]))
total_timesteps = 1000000

callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mybuddy_policy_checkpoint")

model = SAC(
    saccnn,  # You can replace "MlpPolicy" with your custom policy if needed
    my_env,
    verbose=1,
    buffer_size=100000,  # Replay buffer size
    batch_size=256,
    learning_rate=0.00001,
    gamma=0.9,
    ent_coef='auto',
    tau=0.005,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    learning_starts=100,
    use_sde=False,
    device="cuda:0",
    tensorboard_log=f"{log_dir}/tensorboard",
)

model.learn(
    total_timesteps=total_timesteps,
    callback=[WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/models/{run.id}",
        verbose=2), callback]
)

run.finish()

my_env.close()
