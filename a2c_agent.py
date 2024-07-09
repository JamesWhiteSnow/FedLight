import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


tf.keras.backend.set_floatx('float64')


class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.actor_lr = 0.0001
        self.opt = tf.keras.optimizers.Adam(self.actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        return policy_loss

    def upload_gradient(self,states,actions,advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return grads

    def train(self,grads):
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.critic_lr = 0.0002
        self.opt = tf.keras.optimizers.Adam(self.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def upload_gradient(self,states,td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        return grads

    def train(self,grads):
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

class A2CAgent:
    def __init__(self, intersection_id, state_size, action_size):
        self.intersection_id = intersection_id
        self.state_dim = state_size
        self.action_dim = action_size
        self.gamma = 0.95
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.state_batch = []
        self.action_batch = []
        self.td_target_batch = []
        self.advatnage_batch = []

    def choose_action(self, state):
        probs = self.actor.model.predict(
            np.reshape(state, [1, self.state_dim]))
        action = np.random.choice(self.action_dim, p=probs[0])
        return action

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_dim]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def remember(self, state, action, next_state, reward, done):
        state = np.reshape(state, [1, self.state_dim])
        action = np.reshape(action, [1, 1])
        next_state = np.reshape(next_state, [1, self.state_dim])
        reward = np.reshape(reward, [1, 1])

        td_target = self.td_target(reward, next_state, done)
        advantage = self.advatnage(
            td_target, self.critic.model.predict(state))

        self.state_batch.append(state)
        self.action_batch.append(action)
        self.td_target_batch.append(td_target)
        self.advatnage_batch.append(advantage)

    def upload_gradient(self):
        states = self.list_to_batch(self.state_batch)
        actions = self.list_to_batch(self.action_batch)
        td_targets = self.list_to_batch(self.td_target_batch)
        advantages = self.list_to_batch(self.advatnage_batch)
        actor_gradient = self.actor.upload_gradient(states, actions, advantages)
        critic_gradient = self.critic.upload_gradient(states, td_targets)
        return np.array(actor_gradient),np.array(critic_gradient)

    def train(self,actor_gradient,critic_gradient):
        self.actor.train(actor_gradient)
        self.critic.train(critic_gradient)
        self.state_batch = []
        self.action_batch = []
        self.td_target_batch = []
        self.advatnage_batch = []

    def load(self, name):
        self.actor.model.load_weights(name + "a2c-actor.h5", by_name=True)
        self.critic.model.load_weights(name + "a2c-critic.h5", by_name=True)

    def save(self, name):
        self.actor.model.save_weights(name + "a2c-actor.h5")
        self.critic.model.save_weights(name + "a2c-critic.h5")
        print("model saved:{}".format(name))


class MA2CAgent():
    def __init__(self, intersection, state_size, phase_list):
        self.intersection = intersection
        self.agents = {}
        self.make_agents(state_size, phase_list)
        self.update_interval = 5
        self.step=0

    def make_agents(self, state_size, phase_list):
        for id_ in self.intersection:
            self.agents[id_] = A2CAgent(id_, state_size=state_size, action_size=len(phase_list[id_]), )

    def remember(self, state, action, next_state, reward, done):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_], action[id_], next_state[id_], reward[id_], done)

    def choose_action(self, state):
        self.step +=1
        action = {}
        for id_ in self.intersection:
            action[id_] = self.agents[id_].choose_action(state[id_])
        return action

    def train(self, done):
        if self.step >= self.update_interval or done:
            self.step=0
            actor_gradients={}
            critic_gradients={}

            for id_ in self.intersection:
                actor_gradients[id_],critic_gradients[id_]=self.agents[id_].upload_gradient()

            mean_actor_gradient=sum(actor_gradients.values())/len(actor_gradients)
            mean_critic_gradient=sum(critic_gradients.values())/len(critic_gradients)
            for id_ in self.intersection:
                self.agents[id_].train(mean_actor_gradient,mean_critic_gradient)

    def load(self, name):
        for id_ in self.intersection:
            self.agents[id_].load(name)
        print('successfully load model!\n')

    def save(self, name):
        for id_ in self.intersection:
            self.agents[id_].save(name)
