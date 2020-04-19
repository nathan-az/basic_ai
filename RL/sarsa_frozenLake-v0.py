import gym
import numpy as np
import random
import time

env = gym.make("FrozenLake-v0")
epsilon = 0.9
gamma = 0.8
alpha = 0.8
num_episodes = 100
max_path = 100
Q = np.zeros((env.observation_space.n, env.action_space.n))

print(env.observation_space.n)
print(env.action_space.n)


# greedy epsilon algorithm for choosing state
def choose_action(state):
    rand = random.random()
    if rand < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

# update_Q written in prediction - target form
# alternative is (1-alpha)Q(s, a) + alpha(r + gamma(Q(s2, a2)))
# equivalent, but prediction-target implies objective function of squared error
def update_Q(s, a, r, s2, a2):
    Q[s, a] = Q[s, a] + alpha*(r + gamma*Q[s2, a2] - Q[s, a])

# this reward below is NOT the actual "reward" inside the algorithm
# for each step we increment by 1
# in this case it will help us track how long each episode lasts without dying
reward = 0
won = 0
for i in range(num_episodes):
    s = env.reset()
    a = choose_action(s)
    for j in range(max_path):
        env.render()
        s2, r, is_end, info = env.step(a)
        a2 = choose_action(s2)
        update_Q(s, a, r, s2, a2)
        s = s2
        a = a2
        reward += 1
        time.sleep(0.05)
        if r == 1:
            won+=1
            # print("Reached end during episode", i)
        if is_end:
            break

print("Average score per episode:", reward/num_episodes)
print("Reached the end", won, "times")
env.close()