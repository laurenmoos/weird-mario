import os

import math
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

import retro

from baselines.common.atari_wrappers import WarpFrame
from .arguments import get_args

args = get_args()


class SnesDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SnesDiscretizer, self).__init__(env)
        buttons = ["B", "Y", "SELECT", "START", "up", "down", "left", "right", "A", "X", "L", "R"]
        actions = [['right'], ['right', 'A'], ['right', 'B'], ['right', 'Y'], ['A'], ['B'], ['left'], ['left', 'A'],
                   ['left', 'B'], ['left', 'Y'], ['A', 'Y'], ['B', 'Y'], ['down'], ['up'], ['Y', 'up'], ['B', 'up'],
                   ['A', 'up'], ['A', 'Y', 'right'],
                   ['A', 'Y', 'left'], ['B', 'Y', 'right'], ['B', 'Y', 'left'], ['SELECT']]

        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

        if args.skip:
            self.shroom_interval = 50
        else:
            self.shroom_interval = 200

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, done, info = self.env.step()
            else:
                ob, rew, done, info = self.env.step()
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step()
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


MUSHROOM_BOX_ADDR = 8261058
CREAMSICLE_STATE = 22


def add_mushroom(env):
    retro._retro.Memory.assign(env.data.memory, MUSHROOM_BOX_ADDR, "uint8", 1)
    return


def set_powerup_state(env, state):
    retro._retro.Memory.assign(env.data.memory, 8257561, "uint8", state)
    return


class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None, dim=84):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, dim, dim), dtype=np.uint8)

        self.timer = args.episode_length
        self.countdown = 0
        self.multiplier = 0
        self.fresh = True
        self.x = 0
        self.s = 0
        self.code_covered = set()
        self.crashed = False

    def step(self):  # pylint: disable=method-hidden

        if args.autoshroom:

            if self.timer % shroom_interval == 0:
                add_mushroom(self.env)

        if args.weird:

            if self.fresh:
                add_mushroom(self.env)
                set_powerup_state(self.env, CREAMSICLE_STATE)

                self.fresh = False
        else:
            if self.fresh:
                set_powerup_state(self.env, 3)
                self.fresh = False

        # action =  self.env.action_space.sample()
        obs, _, done, info = self.env.step()

        if (info['powerup'] != 22) and (info['powerup'] > 3):
            self.crashed = True

        info['crash'] = int(self.crashed)

        self.timer -= 1

        reward = 0

        if args.reward == 0:

            if info['yoshiCoins'] > self.s:
                reward = 1
                self.s = info['yoshiCoins']


        elif args.reward == 1:

            reward = info['yoshiCoins'] - self.s
            reward *= 0.01
            self.s = info['yoshiCoins']


        elif args.reward == 2:

            trace = info['trace'][:args.rtrace_length]
            line = [x[2] for x in trace]
            for word in line:
                if word not in self.code_covered:
                    self.code_covered.add(word)
                    reward = 1


        elif args.reward == 3:

            trace = info['trace'][:args.rtrace_length]
            line = [x[2] for x in trace]
            for word in line:
                if word not in self.code_covered:
                    self.code_covered.add(word)
                    reward += 1
            reward *= 0.01

        elif args.reward == 4:

            trace = info['trace']
            if len(trace) > args.rtrace_length:
                trace = trace[:args.rtrace_length]
            for step in trace:
                addr = step[0]
                if addr not in self.code_covered:
                    self.code_covered.add(addr)
                    reward += 1
            reward *= 0.01

        elif args.reward == 5:
            reward = math.tanh(info["powerup"] / 10.0)

        elif args.reward == 6:
            reward = info["coins"] - self.s
            self.s = info["coins"]
            reward *= 0.01

        if reward == 0:
            self.countdown += 1
        else:
            self.countdown = 0

        if self.countdown > 200:
            done = True

        if done:
            self.timer = args.episode_length
            self.fresh = True
            self.x = 0
            self.s = 0
            self.countdown = 0
            self.code_covered = set()
            self.crashed = False

        return obs, reward, done, info


LEVELS = [
    'YoshiIsland1.state',
    'YoshiIsland2.state',
    'YoshiIsland3.state',
    'YoshiIsland4.state',
    'Bridges1.state',
    'DonutPlains1.state',
    'DonutPlains2.state',
    'DonutPlains3.state',
    'DonutPlains4.state',
    'DonutPlains5.state',
    'Forest1.state',
    'Forest2.state',
    'Forest3.state',
    'Forest4.state',
    'Forest5.state',
    'VanillaDome1.state',
    'VanillaDome2.state',
    'VanillaDome3.state',
    'VanillaDome4.state',
    'VanillaDome5.state',
    'ChocolateIsland1.state',
    'ChocolateIsland2.state',
    'ChocolateIsland3.state',
    'Bridges2.state'
]


def make_env(level, skip, episode_length, env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():

        if level == 0:
            mrank = rank % len(LEVELS)
        else:
            mrank = level % len(LEVELS)

        env = retro.make(game='SuperMarioWorld-Snes', state=LEVELS[mrank])

        env = SnesDiscretizer(env)
        env = WarpFrame(env)

        # Uncomment to repeat each action for 4 frame -- standard for normal play but not always good for 'exploitation'
        if skip:
            # env = MaxAndSkipEnv(env)
            env = StochasticFrameSkip(env, n=4, stickprob=0.25)

        env = TransposeImage(env, op=[2, 0, 1])
        env = TimeLimit(env, max_episode_steps=episode_length)
        env = ProcessFrameMario(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=True)

        # env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(log_dir, device, env, num_frame_stack=4):

    #unroll configurations
    allow_early_resets, num_processes = env['allow_early_resets'], env['num_processes'],

    gamma, level, episode_length, use_skip = env['gamma'], env['level'], env['episode_length'], env['use_skip']

    for i in range(num_processes):
        envs = [
             make_env(level, use_skip, episode_length, env['name'], env['seed'], i, log_dir, allow_early_resets)
        ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, 4, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1)
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step()
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
            info['bad_transition'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
