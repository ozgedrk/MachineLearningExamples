import gym 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Frozen Lake

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table")
print(qtable)


"""
sol : 0
asagi : 1
sag : 2
yukari : 3

"""
# S1  -> (Action) -> S2



episodes = 1000
alpha = 0.5  # Learning Rate
gamma = 0.9   # Discount Rate

outcomes = []

# training

for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False     # Ajanin basari durumu 
    
    outcomes.append("Failure")
    
    while not done:  # Ajan basarili olana kadar state icerisinde hareket et (Action sec ve uygula)
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        
        new_state, reward, done, info, _ = environment.step(action)

        # Update Q table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state

        if reward:
            outcomes[-1] = "Success"

print("Qtable After Training :")
print(qtable)

plt.bar(range(episodes), outcomes)

## TEST

episodes = 100
nb_success = 0
for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False     # Ajanin basari durumu 
    
    while not done:  # Ajan basarili olana kadar state icerisinde hareket et (Action sec ve uygula)
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        
        new_state, reward, done, info, _ = environment.step(action)

        state = new_state

        nb_success += reward


print("Success Rate: ", 100*nb_success/episodes)
