import gymnasium as gym
from stable_baselines3 import A2C, SAC, TD3, DDPG
import os

# run tensorboard with:
# tensorboard --logdir=logs

algorithms_and_policies = {
    A2C: ['MlpPolicy', 'CnnPolicy', 'MultiInputPolicy'],
    SAC: ['MlpPolicy', 'CnnPolicy', 'MultiInputPolicy'],
    TD3: ['MlpPolicy', 'CnnPolicy'],
    DDPG: ['MlpPolicy', 'CnnPolicy']
}

for algorithm, policies in algorithms_and_policies.items():
    model_name = algorithm.__name__
    for policy in policies:
        print(f"Model: {model_name}\nPolicy: {policy}")
        model_dir = f"models/{model_name}/{policy}"    
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        env = gym.make('CarRacing-v3')
        env.reset()

        model = algorithm(policy, env, verbose=1)

        TIMESTEPS = 10000
        for i in range (1, 11):

            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar = True)
            model.save(f"{model_dir}/{TIMESTEPS * i}")