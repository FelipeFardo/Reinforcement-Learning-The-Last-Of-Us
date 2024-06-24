import thelastofus
from custonsmap import customMap4
import numpy as np
import random
from utils import save_q_table,  load_q_table

env = thelastofus.TheLastOfUsEnv(desc=customMap4)

action_size = env.action_space.n
state_size = env.observation_space.n

total_episodes = 1000
learning_rate = 0.8      
max_steps = 99            
gamma = 0.95         

epsilon = 0.01
max_epsilon = 1.0           
min_epsilon = 0.01        
decay_rate = 0.00005     

rewards = []


qtable = np.zeros((state_size, action_size))

try:
    qtable = load_q_table()
    print("Q-table load success.")
except FileNotFoundError:
    print("No Q-table found, creating a new one.")
print(env.P)

for episode in range(total_episodes):
    state = env.reset()[0]
    step = 0
    done = False
    total_rewards = 0

    tired = step == max_steps
    while not tired and not done:
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info, _ = env.step(action)


        qtable[state,action] = qtable[state,action] + learning_rate*(reward + gamma*np.max(qtable[new_state,:]) - qtable[state,action])
        
        total_rewards += reward
        

        state = new_state

        step+=1
        tired = step == max_steps

    print('episode:', episode,' steps:', step, 'rewards:', total_rewards)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

    save_q_table(qtable)
    rewards.append(total_rewards)

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)



