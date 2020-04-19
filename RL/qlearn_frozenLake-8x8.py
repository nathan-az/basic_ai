import numpy as np
import gym
from time import sleep
import random

# alpha is learning rate in line with text. eta also used
alpha = 0.85
gamma = 0.99
max_steps = 50

env = gym.make("FrozenLake8x8-v0")
# env = gym.make("FrozenLake8x8-v0")
Q = np.zeros((env.observation_space.n, env.action_space.n))


# makes next choice with epsilon greedy
# epsilon variable for ease of test run afterwards
def choose_next(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

# note that r here is what the book calls r(t+1)
# i prefer r or r(t) beacuse it is the reward for taking action a(t) from s(t)
def update(s, a, r, s2, alpha):
    Q[s, a] = Q[s, a] + alpha*(r + gamma*np.max(Q[s2]) - Q[s, a])

def run_q_learn(epsilon, alpha, num_episodes, animate):
    # only counting rewards if reaches end where r==1 in this gym environment
    rewards = 0
    for i in range(num_episodes):
        s = env.reset()
        for j in range(max_steps):
            if animate:
                print("Episode",i,"Step",j)
                env.render()
                sleep(0.02)
            # choose next action basaed on policy
            a = choose_next(s, epsilon)
            s2, r, is_end, info = env.step(a)
            # updates based on optimal following action, independent of action taken
            update(s, a, r, s2, alpha)
            # next step becomes current step
            s = s2
            # for this environment, rewards in (0, 1) and 1 only if goal
            rewards += r
            if is_end:
                break
    print("Rewards of {} in {} episodes:".format(rewards, num_episodes))
    # print(Q)

# training Q with 0.9 learning rate, epsilon==1 for full exploration
run_q_learn(1, 0.9, 20000, False)
# attempt at running test set with full exploitation and very low learning rate
run_q_learn(0.00, 0.1, 100, False)
env.close()
