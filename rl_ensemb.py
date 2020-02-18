import numpy as np           # Handle matrices
import gym
import matplotlib.pyplot as plt # Display graphs
from collections import deque# Ordered collection with ends
import random
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeRegressor

env = gym.make('CartPole-v1')

print("The size of our state is: ", env.observation_space)
print("The action size is : ", env.action_space.n)

### MODEL HYPERPARAMETERS
state_size = 4
action_size = env.action_space.n # 2 possible actions
learning_rate = 0.001     # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50000            # Total episodes for training
max_steps = 10000              # Max possible steps in an episode
batch_size = 200                # Batch size
max_tau = 100                #Tau is the C step where we update our target network

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.1            # minimum exploration probability
decay_rate = 0.0001          # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000          # Number of experiences the Memory can keep


### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


class DQNetwork:
    def __init__(self, actions_states, q_values):
        self.ensemble = deque(maxlen=10)
        dt = DecisionTreeRegressor()
        self.ensemble.append(dt.fit(X=actions_states, y=q_values))

    def add_tree(self, state_actions, target_Qs_batch):
        dt = DecisionTreeRegressor()
        self.ensemble.append(dt.fit(X=state_actions, y=target_Qs_batch))

    def predict_q(self, states):
        if len(states.shape)==1:
            r = 1
        else:
            r = states.shape[0]
        q_values = np.zeros((r, 2))

        for i in range(r):
            state_act = np.zeros((2, 5))
            state_act[0, :4] = states[i]
            state_act[0, 4] = 0
            state_act[1, :4] = states[i]
            state_act[1, 4] = 1
            for tree in self.ensemble:
                q_values[i] += tree.predict(state_act)/len(self.ensemble)

        return q_values





# # Instantiate the DQNetwork
# DQNnetwork = DQNetwork(state_size, action_size, learning_rate, name="DQNetwork")
#
# # Instantiate the target network
# TargetNetwork = DQNetwork(state_size, action_size, learning_rate, name="TargetNetwork")


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()

    # Get the next_state, the rewards, done by taking a random action
    action = random.randrange(action_size)

    next_state, reward, done, _ = env.step(action)

    # If the episode is finished
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        state = env.reset()

    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        # Our new state is now the next_state
        state = next_state




def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, ensemble):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.randrange(action_size)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = ensemble.predict_q(state)


        # Take the biggest Q value (= the best action)
        action = np.argmax(Qs)
    #         action = possible_actions[choice]

    return action, explore_probability

batch = memory.sample(batch_size)
states_mb = np.array([each[0] for each in batch], ndmin=3)
actions_mb = np.array([each[1] for each in batch])
rewards_mb = np.array([each[2] for each in batch])
next_states_mb = np.array([each[3] for each in batch], ndmin=3)
dones_mb = np.array([each[4] for each in batch])
X = np.zeros((batch_size, 5))
X[:, :4] = states_mb[0]
X[:, 4] = actions_mb
Y = np.random.rand(actions_mb.shape[0])*500
ensemble = DQNetwork(X, Y)


rewards_list = []

if training == True:
    # Initialize the decay rate (that will use to reduce epsilon)
    decay_step = 0

    # Set tau = 0
    tau = 0
    for episode in range(total_episodes):
        # Set step to 0
        step = 0

        # Initialize the rewards of the episode
        episode_rewards = []

        # Make a new episode and observe the first state
        state = env.reset()

        while step < max_steps:
            step += 1

            # Increase the C step
            tau += 1

            # Increase decay_step
            decay_step += 1

            # Predict the action to take and take it
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, ensemble)

            # Perform the action and get the next_state, reward, and done information

            next_state, reward, done, _ = env.step(action)

            if episode_render:
                env.render()

            # Add the reward to total reward
            episode_rewards.append(reward)

            # If the game is finished
            if done:
                # The episode ends so no next state
                next_state = np.zeros((4), dtype=np.int)

                # Set step = max_steps to end the episode
                step = max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)
                rewards_list.append(total_reward)
                if len(rewards_list) > 100:
                    average_reward = np.sum(rewards_list[-100:])/100
                else:
                    average_reward = 0
                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Explore P: {:.4f}'.format(explore_probability),
                      'Average reward {:.4f}'.format(average_reward))

                # Store transition <st,at,rt+1,st+1> in memory D
                memory.add((state, action, reward, next_state, done))


            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state

            ### LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])
            target_Qs_batch = []

            state_actions = np.zeros((batch_size, 5))
            state_actions[:, :4] = states_mb[0]
            state_actions[:, 4] = actions_mb

            # Get Q values for next_state
            q_next_state = ensemble.predict_q(next_states_mb[0])

            # # Calculate Qtarget for all actions that state
            # q_target_next_state = ensemble.predict_q(next_states_mb[0])

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]
                # We got a'
                action = np.argmax(q_next_state[i])
                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    # Take the Qtarget for action a'
                    target = rewards_mb[i] + gamma * q_next_state[i][action]
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            if tau > max_tau:
                # Update the parameters of our TargetNetwork with DQN_weights
                ensemble.add_tree(state_actions, targets_mb)
                tau = 0

