from mybuddy_env import MyBuddyEnv as maniEnv
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy as saccnn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import optuna


set_random_seed(42)

def objective(trial):
    # Suggest hyperparameters
    # gamma = trial.suggest_loguniform('gamma', 0.9, 0.9999)
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    # ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    buffer_size = trial.suggest_int('buffer_size', 10000, 200000)
    learning_starts = trial.suggest_int('learning_starts', 100, 1000)
    batch_size = trial.suggest_int('batch_size', 64, 256)
    tau = trial.suggest_loguniform('tau', 0.005, 0.05)
    gamma = trial.suggest_loguniform('gamma', 0.8, 0.9)
    train_freq = trial.suggest_int('train_freq', 1, 10)
    gradient_steps = trial.suggest_int('gradient_steps', 1, 10)

    # Create the environment
    env = my_env

    # Create the model
    model = SAC(saccnn, env, learning_rate=learning_rate, buffer_size=buffer_size,
                learning_starts=learning_starts, batch_size=batch_size, tau=tau,
                gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, verbose=1)

    # Train the model
    model.learn(total_timesteps=200000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward


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

# set headless to false to visualize training
my_env = maniEnv(config=CONFIG)
check_env(my_env)
my_env = Monitor(my_env)

# Create the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

# Print the best hyperparameters
print(study.best_params)
