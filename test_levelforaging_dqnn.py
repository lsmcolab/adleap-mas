#from src.reasoning.estimation import uniform_estimation
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
import sys
import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
import src.envs.LevelForagingEnv
sys.path.append('src/reasoning')

# Define the TF-model.
class MyModel(tf.keras.Model):
    def __init__(self, state_shape, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.shape = state_shape
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_shape[0], state_shape[1], 1))
        self.conv2D = tf.keras.layers.Conv2D(16,kernel_size = (3,3),activation="relu",kernel_initializer="glorot_normal")
        self.pool = tf.keras.layers.MaxPool2D((2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='glorot_normal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='glorot_normal')

    @tf.function
    def call(self, inputs):
        inputs = tf.reshape(inputs,(-1,self.shape[0],self.shape[1],1))
        # Optional : Change as Required .
        inputs = tf.clip_by_value(inputs,-10,10)
        z = self.input_layer(inputs)
        z = self.conv2D(z)
        z = self.pool(z)
        z = self.flatten(z)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        # Maximum and minimum number of experiences. No training occurs before min_experience steps
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences


    def predict(self, inputs):

        return self.model(inputs)

    # Executes one step of gradient descent using sampled batch
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0

        #  Sampling from experience buffer.
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)

        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)


            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            # atleast_2D part is not really needed here.
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    number_of_repeats = 10
    observations = env.reset()
    max_episodes= 50
    state_shape = env.state_set.state_representation.shape  # Or use env.state.shape

    losses = list()
    while not done and env.episode<max_episodes:
    #    action = TrainNet.get_action(observations, epsilon)
        action = TrainNet.get_action(np.reshape(observations.state,(state_shape[0],state_shape[1],1)),epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        #exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        exp = {'s':np.reshape(prev_observations.state,(state_shape[0],state_shape[1],1) )\
            ,'a':action,'r':reward,'s2':np.reshape(observations.state,(state_shape[0],state_shape[1],1)),'done':done}

        # Increase number of repeats to do on-line training
        for i in range(number_of_repeats):
            TrainNet.add_experience(exp)


        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)

def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    agents_color = {'l1': 'lightgrey', 'l2': 'darkred', 'l3': 'darkgreen', 'l4': 'darkblue', 'entropy': 'blue',
                    'mcts': 'yellow'}
    while not done and env.episode<10:
        env.render(agents_color)
        action = TrainNet.get_action(np.reshape(observation.state,(10,10,1)), 0)

        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    components = {
        'agents': [

            Agent(index='A', atype='mcts', position=(0, 0), direction=np.pi / 2, radius=1.0, angle=1.0, level=1.0),
            Agent(index='B', atype='l1', position=(0, 9), direction=np.pi / 2, radius=0.7, angle=5.0, level=0.3),
            Agent(index='C', atype='l2', position=(9, 9), direction=np.pi / 2, radius=0.5, angle=3.0, level=0.3),
            Agent(index='D', atype='l3', position=(9, 0), direction=np.pi / 2, radius=0.6, angle=7.0, level=0.3),
        ],
        'adhoc_agent_index': 'A',
        'tasks': [Task('1', (2, 2), 1.0),
                  Task('2', (4, 4), 1.0),
                  Task('3', (5, 5), 1.0),
                  Task('4', (8, 8), 1.0)]}


    env = LevelForagingEnv((10, 10), components)
    gamma = 0.99  # Discount Factor
    copy_step = 25
    #num_states = len(env.observation_space.sample())
    state_shape = env.state_set.state_representation.shape # TODO: Simplify this
    num_actions = env.action_space.n
    hidden_units = [20, 20]
    max_experiences = 1000
    min_experiences = 100
    batch_size = 32
    lr = 1e-5
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'results/dqn_' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(state_shape, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(state_shape, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    # Increase this for proper training
    N = 500
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(0,N):
        env.reset()
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("avg reward for last 100 episodes:", avg_rewards)
    env.close()

    make_video(env, TrainNet)
    env.close()


if __name__ == '__main__':
    for i in range(1):
        main()