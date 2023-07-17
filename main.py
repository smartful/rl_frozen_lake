import gymnasium as gym
import numpy as np
from dynamic_prog_functions import evaluatePolicy, improvePolicy
from gymnasium.spaces.utils import flatdim

# Loading Frozen Lake environment
env = gym.make('FrozenLake-v1', map_name="4x4",
               is_slippery=True, render_mode="human")

# Initialisation
env.nA = flatdim(env.action_space)
env.nS = flatdim(env.observation_space)
state, info = env.reset(seed=42)
pi = np.ones([env.nS, env.nA]) * 0.5
V = np.zeros([env.nS, 1])
gamma = 0.99
k = 5

# Optimization of the policy
i = 0
while True:
    i += 1
    V, improved = evaluatePolicy(env, pi, V, gamma, k)
    pi = improvePolicy(env, pi, V, gamma)

    if (improved == False):
        print(f"Terminé après {str(i)} itérations.")
        break

for _ in range(200):
    action = np.argmax(pi[state])  # this is where you would insert your policy
    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        state, info = env.reset()

env.close()
