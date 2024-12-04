import gymnasium as gym
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.box2d.car_racing import CarRacing

class CustomCarRacing(CarRacing):
    def __init__(self, render_mode = None):
        super().__init__(render_mode=render_mode)
        self.wind_force = 0.1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)  
        return self.step(None)[0], {}
    
    def step(self, action):
        if action is None:
            action = [0.0, 0.0, 0.0]

        action[0] += self.wind_force * np.random.uniform(-1,1)
        action = np.clip(action,-1,1)
        obs, reward, done, truncated, info = super().step(action)
        
        if self.tile_visited_count < 0.9 * len(self.track):
            reward -= 1
    
        return obs, reward, done, truncated, info

if __name__ == "__main__":
    env = CustomCarRacing(render_mode = "rgb_array")
    check_env(env)
    obs = env.reset()

    for _ in range(100):
        action = env.action_space.sample() # random actions
        obs, reward, done, truncated, info = env.step(action) # excutes and action
        if done or truncated:
            obs = env.reset()
    env.close()