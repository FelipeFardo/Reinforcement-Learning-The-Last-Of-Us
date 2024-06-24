import thelastofus
from custonsmap import customMap4
import numpy as np
import random
from utils import load_q_table

env = thelastofus.TheLastOfUsEnv(desc=customMap4, render_mode="human")

action_size = env.action_space.n
state_size = env.observation_space.n

max_steps = 99            
            
rewards = []


qtable = []

try:
    qtable = load_q_table()
    print("Q-table load success.")
except FileNotFoundError:
    print("No Q-table found, creating a new one.")
print(qtable)


env.reset()

for episode in range(5):
    state = env.reset()[0]
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info, _ = env.step(action)
        
        if done:

            env.render()
            
            print("Number of steps", step)
            break
        state = new_state
env.close()