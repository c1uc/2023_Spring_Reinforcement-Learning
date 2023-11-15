import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import math
import scipy.optimize
from collections import namedtuple
from datetime import datetime
import numpy as np
import random
import gym


writer = SummaryWriter("./tb_record_simhash_continuous")
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


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        state_values = self.value_head(x)
        return state_values


class TRPO(object):
    def __init__(self, env, hidden_size=32, gamma=0.995, k=32):
        self.env = env
        self.hidden_size = hidden_size
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.simhash = SimHash(k, self.num_states, 0.01)

        self.actor = Actor(num_inputs=self.num_states, num_outputs=self.num_actions)

        self.critic = Critic(num_inputs=self.num_states)

        self.gamma = gamma
        self.memory = []

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.actor(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action

    @staticmethod
    def set_flat_params_to(model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    @staticmethod
    def get_flat_params_from(model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))

        flat_params = torch.cat(params)
        return flat_params

    @staticmethod
    def get_flat_grad_from(net, grad_grad=False):
        grads = []
        for param in net.parameters():
            if grad_grad:
                grads.append(param.grad.grad.view(-1))
            else:
                grads.append(param.grad.view(-1))

        flat_grad = torch.cat(grads)
        return flat_grad

    @staticmethod
    def normal_log_density(x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (
                2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)

    def linesearch(self, model,
                   f,
                   x,
                   fullstep,
                   expected_improve_rate,
                   max_backtracks=10,
                   accept_ratio=.1):
        fval = f(True).data
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            self.set_flat_params_to(model, xnew)
            newfval = f(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew
        return False, x

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
        advantages = (returns - baselines).detach().flatten()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            self.set_flat_params_to(self.critic, torch.Tensor(flat_params))
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.critic(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum() * 1e-3
            value_loss.backward()
            return value_loss.data.double().numpy(), self.get_flat_grad_from(self.critic).data.double().numpy()

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                                self.get_flat_params_from(self.critic).double().numpy(),
                                                                maxiter=25)
        self.set_flat_params_to(self.critic, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = self.actor(Variable(states))
        fixed_log_prob = self.normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.actor(Variable(states))
            else:
                action_means, action_log_stds, action_stds = self.actor(Variable(states))

            log_prob = self.normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        def get_kl():
            mean1, log_std1, std1 = self.actor(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        self.trpo_step(self.actor, get_loss, get_kl)

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

    def trpo_step(self, model, get_loss, get_kl, max_kl=1e-2, damping=1e-1):
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

        stepdir = self.conjugate_gradient(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

        prev_params = self.get_flat_params_from(model)
        success, new_params = self.linesearch(model, get_loss, prev_params, fullstep,
                                         neggdotstepdir / lm[0])
        self.set_flat_params_to(model, new_params)

        return loss

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
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = self.env.step(action)
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
                action = action.data[0].numpy()
                next_state, _, done, _ = self.env.step(action)
                state = next_state
                if done:
                    break


if __name__ == '__main__':
    random_seed = 10
    gym.envs.register(
        id='MyMountainCar',
        entry_point='gym.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv',
        max_episode_steps=500
    )
    test_env = gym.make('MountainCarContinuous-v0')
    test_env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    agent = TRPO(test_env, hidden_size=32, gamma=0.99)
    agent.train(num_epoch=1000, update_step=5000, show_freq=50)
    agent.eval(num_episode=5)

