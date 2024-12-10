import gymnasium as gym
from stable_baselines3 import PPO
import os

# run tensorboard with:
# tensorboard --logdir=logs

models_dir = "models1/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('CarRacing-v3', render_mode = "rgb_array")
env.reset()

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 50000


for i in range(10):
    print(f"==== Episode {i} ====")

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name= "PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:

        action, _states = model.predict(obs)
        obs, reward, trunc, done, info = env.step(action)
        total_reward += reward
        env.render()
   
    print(f"Episode {i} finished with a total reward of: {total_reward}")
    print("=====================")