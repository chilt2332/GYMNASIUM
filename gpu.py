import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
import os

# Ensure TensorBoard logging directory exists
tensorboard_log_dir = "logs2"
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Check for GPU and set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom callback for additional logging
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Custom logic (e.g., logging extra information) can be added here
        return True

# Define algorithms and policies
algorithms_and_policies = {
    "TD3": [TD3, ['CnnPolicy','MlpPolicy']],
}

# Training loop
for algorithm_name, (algorithm_class, policies) in algorithms_and_policies.items():
    for policy in policies:
        print(f"Starting training for {algorithm_name} with {policy}")

        # Create model directory
        model_dir = f"models2/{algorithm_name}/{policy}"
        os.makedirs(model_dir, exist_ok=True)

        # Create environment
        env = gym.make('CarRacing-v3')
        env.reset()

        # Configure the logger for TensorBoard
        log_path = f"{tensorboard_log_dir}/{algorithm_name}_{policy}"
        new_logger = configure(log_path, ["stdout", "tensorboard"])

        # Initialize the model
        if algorithm_class in [SAC, TD3, DDPG]:
            model = algorithm_class(policy, env, verbose=1, buffer_size=10000, tensorboard_log=log_path, device=device)
        else:
            model = algorithm_class(policy, env, verbose=1, tensorboard_log=log_path, device=device)

        # Set the custom logger
        model.set_logger(new_logger)

        # Training parameters
        TIMESTEPS = 10000
        NUM_ITERATIONS = 10

        for i in range(1, NUM_ITERATIONS + 1):
            print(f"Training iteration {i}/{NUM_ITERATIONS} for {algorithm_name} with {policy}")
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=CustomCallback())

            # Save model
            model.save(f"{model_dir}/{TIMESTEPS * i}")

        print(f"Completed training for {algorithm_name} with {policy}")
        break
    break

print("All trainings completed. Run 'tensorboard --logdir=logs' to view results.")
