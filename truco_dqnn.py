from src.envs.TrucoEnv import TrucoEnv
from src.reasoning.estimation import truco_uniform_estimation
import sys
import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
sys.path.append('src/reasoning')

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
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
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
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


def play_game(env, TrainNet, TargetNet, epsilon, copy_step,main_player="None"):
    rewards = 0
    iter = 0
    number_of_repeats = 1
    state = env.reset()
    losses = list()
    while env.state['points'][0] < 12 and env.state['points'][1] < 12:

        info, done = None, False
        while not done:
            # Rendering the environment

            # Agent taking an action
            current_player = env.get_adhoc_agent()
            current_player.next_action = -1
            action = TrainNet.get_action(state.feature(),epsilon)
            while None in current_player.hand[action]:
                action = TrainNet.get_action(state.feature(),
                                             epsilon)


                # Step on environment

            prev_state = state
            state, reward, done, info = env.step(action)
            rewards+=reward
            exp = {'s': prev_state.feature(), 'a':action, 'r' : reward, 's2' : state.feature(), 'done' : done }
            # Verifying the end condition

            for i in range(number_of_repeats):
                if(main_player == "None" or current_player.index == main_player):
                    TrainNet.add_experience(exp)

            loss = TrainNet.train(TargetNet)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())
            iter += 1
            if iter % copy_step == 0:
                TargetNet.copy_weights(TrainNet)


            if done:
                break

        env.deal()
        state = env.get_observation()


    return rewards, mean(losses)



# def make_video(env, TrainNet):
#     env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
#     rewards = 0
#     steps = 0
#     done = False
#     observation = env.reset()
#     while not done:
#         env.render()
#         action = TrainNet.get_action(observation, 0)
#         observation, reward, done, _ = env.step(action)
#         steps += 1
#         rewards += reward
#     print("Testing steps: {} rewards {}: ".format(steps, rewards))



def main():
    env = TrucoEnv(players=['MATHEUS', 'LEANDRO', 'AMOKH', 'YEHIA'],
                   reasoning=['mcts', 'pomcp', 't2', 't1'])
    main_player = "MATHEUS"
    gamma = 0.99
    copy_step = 25
    state_shape = 18
    num_actions = env.action_space.n
    hidden_units = [20, 20]
    max_experiences = 500
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'results/Truco_dqn_' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(state_shape, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(state_shape, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 500
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step,main_player)
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


if __name__ == '__main__':
    for i in range(1):
        main()