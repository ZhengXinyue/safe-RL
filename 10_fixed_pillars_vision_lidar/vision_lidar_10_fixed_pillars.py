import random
from collections import deque
import time
import os

import gym
import safety_gym
from safety_gym.envs.engine import Engine
import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss, nn
import gluonbook as gb


class MemoryBuffer:
    def __init__(self, buffer_size, ctx):
        self.buffer = deque(maxlen=buffer_size)
        self.maxsize = buffer_size
        self.ctx = ctx

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        assert len(self.buffer) > batch_size
        minibatch = random.sample(self.buffer, batch_size)
        vision_batch = nd.array([data[0][0] for data in minibatch], ctx=self.ctx)
        lidar_batch = nd.array([data[0][1] for data in minibatch], ctx=self.ctx)
        action_batch = nd.array([data[1] for data in minibatch], ctx=self.ctx)
        reward_batch = nd.array([data[2] for data in minibatch], ctx=self.ctx)
        next_vision_batch = nd.array([data[3][0] for data in minibatch], ctx=self.ctx)
        next_lidar_batch = nd.array([data[3][1] for data in minibatch], ctx=self.ctx)
        done = nd.array([data[4] for data in minibatch], ctx=self.ctx)
        return vision_batch, lidar_batch, action_batch, reward_batch, next_vision_batch, next_lidar_batch, done

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)


class Actor(nn.Block):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim

        self.conv0 = nn.Conv2D(32, kernel_size=8, strides=4, padding=2, activation='relu')
        self.conv1 = nn.Conv2D(64, kernel_size=4, strides=2, padding=1, activation='relu')
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation='relu')

        self.dense0 = nn.Dense(512, activation='relu')
        self.dense1 = nn.Dense(256, activation='relu')
        self.dense2 = nn.Dense(128, activation='relu')
        self.dense3 = nn.Dense(self.action_dim, activation='tanh')

    def forward(self, vision_state, lidar_state):
        feature = self.conv2(self.conv1(self.conv0(vision_state))).flatten()
        dense_input = nd.concat(feature, lidar_state, dim=1)
        action = self.dense3(self.dense2(self.dense1(self.dense0(dense_input))))
        return action


class Critic(nn.Block):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv0 = nn.Conv2D(32, kernel_size=8, strides=4, padding=2, activation='relu')
        self.conv1 = nn.Conv2D(64, kernel_size=4, strides=2, padding=1, activation='relu')
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation='relu')

        self.dense0 = nn.Dense(512, activation='relu')
        self.dense1 = nn.Dense(256, activation='relu')
        self.dense2 = nn.Dense(128, activation='relu')
        self.dense3 = nn.Dense(1)

    def forward(self, vision_state, lidar_state, action):
        feature = self.conv2(self.conv1(self.conv0(vision_state))).flatten()
        dense_input = nd.concat(feature, lidar_state, action, dim=1)
        q_value = self.dense3(self.dense2(self.dense1(self.dense0(dense_input))))
        return q_value


class TD3:
    def __init__(self,
                 action_dim,
                 actor_learning_rate,
                 critic_learning_rate,
                 batch_size,
                 memory_size,
                 gamma,
                 tau,
                 explore_steps,
                 policy_update,
                 policy_noise,
                 explore_noise,
                 noise_clip,
                 ctx):
        self.action_dim = action_dim

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.explore_steps = explore_steps
        self.policy_update = policy_update
        self.policy_noise = policy_noise
        self.explore_noise = explore_noise
        self.noise_clip = noise_clip
        self.ctx = ctx

        self.main_actor_network = Actor(action_dim)
        self.target_actor_network = Actor(action_dim)
        self.main_critic_network1 = Critic()
        self.target_critic_network1 = Critic()
        self.main_critic_network2 = Critic()
        self.target_critic_network2 = Critic()

        self.main_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_actor_network.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network1.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network1.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.main_critic_network2.collect_params().initialize(init=init.Xavier(), ctx=ctx)
        self.target_critic_network2.collect_params().initialize(init=init.Xavier(), ctx=ctx)

        self.actor_optimizer = gluon.Trainer(self.main_actor_network.collect_params(),
                                             'adam',
                                             {'learning_rate': self.actor_learning_rate})
        self.critic1_optimizer = gluon.Trainer(self.main_critic_network1.collect_params(),
                                               'adam',
                                               {'learning_rate': self.critic_learning_rate})
        self.critic2_optimizer = gluon.Trainer(self.main_critic_network2.collect_params(),
                                               'adam',
                                               {'learning_rate': self.critic_learning_rate})

        self.total_steps = 0
        self.total_train_steps = 0

        self.memory_buffer = MemoryBuffer(buffer_size=self.memory_size, ctx=ctx)

    def choose_action_train(self, state):
        visual = nd.array([state[0]], ctx=self.ctx)
        lidar = nd.array([state[1]], ctx=self.ctx).flatten()
        action = self.main_actor_network(visual, lidar)
        # no noise clip
        noise = nd.normal(loc=0, scale=self.explore_noise, shape=action.shape, ctx=self.ctx)
        action += noise
        clipped_action = self.action_clip(action).squeeze()
        return clipped_action

    def choose_action_evaluate(self, state):
        visual = nd.array([state[0]], ctx=self.ctx)
        lidar = nd.array([state[1]], ctx=self.ctx).flatten()
        action = self.main_actor_network(visual, lidar)
        return action

    def action_clip(self, action):
        clipped_action = nd.clip(action, -1, 1)
        return clipped_action

    def soft_update(self, target_network, main_network):
        target_parameters = target_network.collect_params().keys()
        main_parameters = main_network.collect_params().keys()
        d = zip(target_parameters, main_parameters)
        for x, y in d:
            target_network.collect_params()[x].data()[:] = \
                target_network.collect_params()[x].data() * \
                (1 - self.tau) + main_network.collect_params()[y].data() * self.tau

    def update(self):
        self.total_train_steps += 1
        vision_batch, lidar_batch, action_batch, reward_batch, \
        next_vision_batch, next_lidar_batch, done_batch = self.memory_buffer.sample(self.batch_size)

        # --------------optimize the critic network--------------------
        with autograd.record():
            # choose next action according to target policy network
            next_action_batch = self.target_actor_network(next_vision_batch, next_lidar_batch)
            noise = nd.normal(loc=0, scale=self.policy_noise, shape=next_action_batch.shape, ctx=self.ctx)
            # with noise clip
            noise = nd.clip(noise, a_min=-self.noise_clip, a_max=self.noise_clip)
            next_action_batch = next_action_batch + noise
            clipped_action = self.action_clip(next_action_batch)

            # get target q value
            target_q_value1 = self.target_critic_network1(next_vision_batch, next_lidar_batch, clipped_action)
            target_q_value2 = self.target_critic_network2(next_vision_batch, next_lidar_batch, clipped_action)
            target_q_value = nd.minimum(target_q_value1, target_q_value2).squeeze()
            target_q_value = reward_batch + (1.0 - done_batch) * (self.gamma * target_q_value)

            # get current q value
            current_q_value1 = self.main_critic_network1(vision_batch, lidar_batch, action_batch)
            current_q_value2 = self.main_critic_network2(vision_batch, lidar_batch, action_batch)
            loss = gloss.L2Loss()

            value_loss1 = loss(current_q_value1, target_q_value.detach())
            value_loss2 = loss(current_q_value2, target_q_value.detach())

        self.main_critic_network1.collect_params().zero_grad()
        self.main_critic_network2.collect_params().zero_grad()

        autograd.backward([value_loss1, value_loss2])

        self.critic1_optimizer.step(self.batch_size)
        self.critic2_optimizer.step(self.batch_size)

        # ---------------optimize the actor network-------------------------
        if self.total_train_steps % self.policy_update == 0:
            with autograd.record():
                pred_action_batch = self.main_actor_network(vision_batch, lidar_batch)
                actor_loss = -nd.mean(self.main_critic_network1(vision_batch, lidar_batch, pred_action_batch))

            self.main_actor_network.collect_params().zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step(1)

            self.soft_update(self.target_actor_network, self.main_actor_network)
            self.soft_update(self.target_critic_network1, self.main_critic_network1)
            self.soft_update(self.target_critic_network2, self.main_critic_network2)

    def save(self, _time):
        self.main_actor_network.save_parameters('%s/TD3_main_actor_network_at_steps_%d.params' % (_time, self.total_steps))
        self.target_actor_network.save_parameters('%s/TD3_target_actor_network_at_steps_%d.params' % (_time, self.total_steps))
        self.main_critic_network1.save_parameters('%s/TD3_main_critic_network1_at_steps_%d.params' % (_time, self.total_steps))
        self.main_critic_network2.save_parameters('%s/TD3_main_critic_network2_at_steps_%d.params' % (_time, self.total_steps))
        self.target_critic_network1.save_parameters('%s/TD3_target_critic_network1_at_steps_%d.params' % (_time, self.total_steps))
        self.target_critic_network2.save_parameters('%s/TD3_target_critic_network2_at_steps_%d.params' % (_time, self.total_steps))

    def load(self, _time, steps):
        self.main_actor_network.load_parameters('%s/TD3_main_actor_network_at_steps_%d.params' % (_time, steps))
        self.target_actor_network.load_parameters('%s/TD3_target_actor_network_at_steps_%d.params' % (_time, steps))
        self.main_critic_network1.load_parameters('%s/TD3_main_critic_network1_at_steps_%d.params' % (_time, steps))
        self.main_critic_network2.load_parameters('%s/TD3_main_critic_network2_at_steps_%d.params' % (_time, steps))
        self.target_critic_network1.load_parameters('%s/TD3_target_critic_network1_at_steps_%d.params' % (_time, steps))
        self.target_critic_network2.load_parameters('%s/TD3_target_critic_network2_at_steps_%d.params' % (_time, steps))


def plot(result):
    l = []
    for i in range(0, len(result)-100, 20):
        l.append(sum(result[i:i+100]) / 100)
    return l


def f():
    flag = False
    while not flag:
        flag = True
        m = np.random.uniform(-4, 4, size=(10, 2))
        for i in range(len(m)):
            for j in range(i+1, len(m)):
                if np.sqrt(np.sum(np.square(m[i] - m[j]))) < 0.6:
                    flag = False
    l = []
    for i in m:
        l.append(list(i))
    return l


def main():
    config = {
        'num_steps': 2000,
        'robot_base': 'xmls/car.xml',
        'task': 'goal',
        'placements_extents': [-4, -4, 4, 4],
        'observation_flatten': False,
        'sensors_obs': [],

        'observe_pillars': True,
        'observe_vision': True,
        'vision_size': (84, 84),

        'constrain_pillars': True,

        'reward_distance': 0.8,  # if reward_distance is 0, then the reward function is sparse

        'lidar_max_dist': 8,
        'lidar_num_bins': 120,

        'pillars_num': 10,
        'pillars_locations': [[ 3.52087172, -3.29627743],
       [-3.60993032,  3.76972334],
       [ 0.40512185, -1.11828606],
       [-2.41838798,  0.82712593],
       [-3.68081797, -2.80101914],
       [ 1.32806649, -1.6491154 ],
       [-1.33992617,  2.2909115 ],
       [ 1.05154836,  2.05283816],
       [-1.4235473 , -1.75820755],
       [-2.50183935,  3.20091906],
       ]
    }

    env = Engine(config)
    seed = 25437624
    env.seed(seed)
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ctx = gb.try_gpu()
    ctx = mx.cpu()
    max_episodes = 1000
    max_episode_steps = 1000
    _time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    agent = TD3(action_dim=int(env.action_space.shape[0]),
                actor_learning_rate=0.0001,
                critic_learning_rate=0.0001,
                batch_size=64,
                memory_size=1000000,
                gamma=0.99,
                tau=0.005,
                explore_steps=10000,
                policy_update=2,
                policy_noise=0.2,
                explore_noise=0.1,
                noise_clip=0.5,
                ctx=ctx)
    mode = 'train'
    if mode == 'train':
        os.mkdir('%s' % _time)
        render = False
        episode = 0
        while agent.total_steps < 300002:
            episode += 1
            state = env.reset()
            vision = state['vision'].transpose((2, 0, 1))
            lidar = state['pillars_lidar']
            state = [vision, lidar]
            flag = False
            for step in range(max_episode_steps):
                if render:
                    env.render()
                if agent.total_steps < agent.explore_steps:
                    action = env.action_space.sample()
                    agent.total_steps += 1
                else:
                    action = agent.choose_action_train(state)
                    action = action.asnumpy()
                    agent.total_steps += 1
                next_state, reward, done, info = env.step(action)
                next_vision = next_state['vision'].transpose((2, 0, 1))
                next_lidar = next_state['pillars_lidar']
                next_state = [next_vision, next_lidar]
                if agent.total_steps % 20000 == 0:
                    agent.save(_time)
                if info['cost_pillars'] != 0:
                    reward -= 1
                    done = True
                if 'goal_met' in info.keys():
                    reward += 1
                    done = True
                    flag = True
                agent.memory_buffer.store_transition(state, action, reward, next_state, done)
                state = next_state
                if agent.total_steps > agent.explore_steps:
                    agent.update()
                if done:
                    break
            if not flag:
                print('episode %d fails at steps %d' % (episode, agent.total_steps))
            if flag:
                print('episode %d succeeds at steps %d' % (episode, agent.total_steps))

    elif mode == 'test':
        t = '2019-12-02 10:59:40'
        steps = 100000
        agent.load(t, steps)
        render = False
        episode = 0
        success_list = []
        while agent.total_steps < 1000002:
            episode += 1
            state = env.reset()
            vision = state['vision'].transpose((2, 0, 1))
            lidar = state['pillars_lidar']
            state = [vision, lidar]
            flag = True
            for step in range(max_episode_steps):
                if render:
                    env.render()
                action = agent.choose_action_train(state).asnunpy()
                agent.total_steps += 1
                next_state, reward, done, info = env.step(action)
                next_vision = next_state['vision'].transpose((2, 0, 1))
                next_lidar = next_state['pillars_lidar']
                next_state = [next_vision, next_lidar]
                if info['cost_pillars'] != 0:
                    reward -= 1
                    done = True
                if 'goal_met' in info.keys():
                    reward += 1
                    done = True
                    flag = True
                state = next_state
                if done:
                    break
            if not flag:
                success_list.append(0)
                print('episode %d fails at steps %d' % (episode, agent.total_steps))
            if flag:
                success_list.append(1)
                print('episode %d succeeds at steps %d' % (episode, agent.total_steps))
        print(success_list)
    else:
            print('Wrong input')


if __name__ == '__main__':
    main()
