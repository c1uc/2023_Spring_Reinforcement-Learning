import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple
from datetime import datetime
import numpy as np
import random
import gym


writer = SummaryWriter("./tb_record_simhash")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'key'))


class SimHash(object):
    def __init__(self, k, D, beta):
        self.k = k
        self.D = D
        self.beta = beta
        self.A = np.random.normal(0, 1, (k, D))
        self.hash_table = {}
        self.new_hash_table = {}

    def get_keys(self, states):
        keys = []
        for state in states:
            key = (np.asarray(np.sign(self.A @ state), dtype=int) + 1) // 2  # to binary code array
            key = int(''.join(key.astype(str).tolist()), base=2)  # to int (binary)
            keys.append(key)
            if key in self.hash_table:
                self.hash_table[key] += 1
            else:
                self.hash_table[key] = 1
        return np.asarray(keys)

    def get_bonus(self, keys):
        cnt = np.array([self.hash_table[key] for key in keys])
        return self.beta * np.reciprocal(np.sqrt(cnt))


class TRPO(object):
    def __init__(self, env, hidden_size=32, gamma=0.995, lr_c=0.005, k=32):
        self.env = env
        self.hidden_size = hidden_size
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.simhash = SimHash(k, self.num_states, 0.01)

        self.actor = nn.Sequential(nn.Linear(self.num_states, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.num_actions),
                                   nn.Softmax(dim=1))

        self.critic = nn.Sequential(nn.Linear(self.num_states, 1))

        self.critic_loss_func = nn.MSELoss()
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_c)
        self.gamma = gamma
        self.memory = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0)
            dist = Categorical(self.actor(state))
            return dist.sample().item()

    def update_agent(self, delta=0.01, damping=1e-1):
        states = torch.cat([tr.state for tr in self.memory], dim=0).float()
        actions = torch.cat([tr.action for tr in self.memory], dim=0).flatten()

        returns = []
        for tr in self.memory:
            R = 0
            tr_returns = []
            rewards = tr.reward + self.simhash.get_bonus(tr.key)
            for reward in rewards[::-1]:
                R = reward + self.gamma * R
                tr_returns.append(R)
            tr_returns = torch.as_tensor(tr_returns[::-1]).unsqueeze(1)
            returns.append(tr_returns)

        returns = torch.cat(returns, dim=0).float()
        baselines = self.critic(states)

        self.critic_optimizer.zero_grad()
        value_loss = self.critic_loss_func(baselines, returns)
        value_loss.backward()
        self.critic_optimizer.step()

        dist = self.actor(states)
        prob = dist[range(dist.shape[0]), actions]
        const_dist = dist.detach().clone()
        const_prob = prob.detach().clone()

        parameters = list(self.actor.parameters())
        advantages = (returns - baselines).detach().flatten()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        L = ((prob / const_prob) * advantages).mean()
        dL = torch.autograd.grad(L, parameters, retain_graph=True)
        loss_grad = torch.cat([grad.flatten() for grad in dL])

        def Fvp(v):
            kl = self.get_kl(const_dist, dist).mean()
            grads = torch.autograd.grad(kl, parameters, create_graph=True, retain_graph=True)
            flat_grad_kl = torch.cat([grad.flatten() for grad in grads])
            v_v = v.detach().clone()
            kl_v = (flat_grad_kl * v_v).sum()
            grads = torch.autograd.grad(kl_v, parameters, retain_graph=True)
            flat_grad_grad_kl = torch.cat([grad.flatten() for grad in grads]).data
            return flat_grad_grad_kl + v * damping

        stepdir = self.conjugate_gradient(Fvp, loss_grad, 10)
        max_length = torch.sqrt(2 * delta / (stepdir @ Fvp(stepdir)))
        max_step = max_length * stepdir

        def criterion(step):
            self.update_actor(step)
            with torch.no_grad():
                dist_new = self.actor(states)
                prob_new = dist_new[range(dist_new.shape[0]), actions]
                L_new = ((prob_new / const_prob) * advantages).mean()
                KL_new = self.get_kl(const_dist, dist_new).mean()
                if L_new - L > 0 and KL_new <= delta:
                    return True
            self.update_actor(-step)
            return False

        i = 0
        while not criterion((0.5 ** i) * max_step) and i < 10:
            i += 1

    def update_actor(self, grad_flattened):
        n = 0
        for params in self.actor.parameters():
            num_element = params.numel()
            g = grad_flattened[n:n+num_element].view(params.shape)
            params.data += g
            n += num_element

    def conjugate_gradient(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def get_kl(self, p, q):
        return (p * (p.log() - q.log())).sum(-1)

    def train(self, num_epoch=500, update_step=10000, show_freq=None):
        i_episode = 0
        for i in range(num_epoch):
            if show_freq is not None and i % show_freq == 0:
                self.eval(num_episode=1)

            print('Epoch {}/{}'.format(i+1, num_epoch))
            start_time = float(datetime.now().timestamp())
            epoch_rewards = []

            epoch_t = 0
            while True:  # episodes loop
                state = self.env.reset()
                episode_reward = 0
                sample = []

                t = 0
                while True:
                    action = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    reward = 1 if (done and t < 499) else 0
                    sample.append((state, action, reward))
                    episode_reward += reward
                    state = next_state
                    epoch_t += 1
                    t += 1
                    if done:
                        break

                states, actions, rewards = zip(*sample)
                keys = self.simhash.get_keys(states)
                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                actions = torch.as_tensor(actions).unsqueeze(1)
                rewards = np.array(rewards)

                self.memory.append(Transition(states, actions, rewards, keys))
                epoch_rewards.append(episode_reward)
                sample.clear()

                i_episode += 1
                print('Collecting Trajectories...({}/{})\r'.format(epoch_t, update_step), end='')
                tags = ['{}/episode-reward'.format(self.env.unwrapped.spec.id)]
                for tag, value in zip(tags, [episode_reward]):
                    writer.add_scalar(tag, value, i_episode)

                if epoch_t >= update_step:
                    break

            end_time = float(datetime.now().timestamp())
            running_time = end_time - start_time
            print('{}/{} [====================] '.format(epoch_t, epoch_t), end='')
            print('- {:.2f}s {:.2f}ms/step '.format(running_time, running_time * 1000 / epoch_t, 2), end='')
            print('- num_episode: {} - avg_reward: {:.2f}'.format(len(epoch_rewards), sum(epoch_rewards)/len(epoch_rewards)))

            epoch_rewards.clear()
            self.update_agent()
            self.memory.clear()

    def eval(self, num_episode):
        for i in range(num_episode):
            state = self.env.reset()
            while True:
                self.env.render()
                action = self.select_action(state)
                next_state, _, done, _ = self.env.step(action)
                state = next_state
                if done:
                    break


if __name__ == '__main__':
    random_seed = 10
    gym.envs.register(
        id='MyMountainCar',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=500
    )
    test_env = gym.make('MyMountainCar')
    test_env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    agent = TRPO(test_env, hidden_size=32, gamma=0.99, lr_c=0.05)
    agent.train(num_epoch=1000, update_step=5000, show_freq=50)
    agent.eval(num_episode=5)

