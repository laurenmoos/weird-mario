from .wrappers import *

import retro

from baselines.common.atari_wrappers import WarpFrame
from arguments import Config, Environment

import logging

log = logging.getLogger('Environment')

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


def make_env(rank):
    def _thunk():
        config = Config.instance()
        env_config = config.environment

        if env_config[Environment.LEVEL] == 0:
            mrank = rank % len(LEVELS)
        else:
            mrank = env_config[Environment.LEVEL] % len(LEVELS)

        env = retro.make(game='SuperMarioWorld-Snes', state=LEVELS[mrank])

        env = SnesDiscretizer(env)
        env = WarpFrame(env)

        if env_config[Environment.USE_SKIP]:
            # env = MaxAndSkipEnv(env)
            env = StochasticFrameSkip(env, n=4, stickprob=0.25)

        env = TransposeImage(env, op=[2, 0, 1])
        env = TimeLimit(env, max_episode_steps=env_config[Environment.EPISODE_LENGTH])
        env = ProcessFrameMario(env)

        return env

    return _thunk


def make_vec_envs(device, num_frame_stack=4):
    config = Config.instance()
    env_config = config.environment

    num_processes = env_config[Environment.NUM_PROCESSES]

    envs = []
    for i in range(num_processes):
        envs = [
            make_env(rank=i)
            for i in range(num_processes)
        ]

    if num_processes > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        # simple environment for running without multi-threading
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if config[Environment.GAMMA]:
            envs = VecNormalize(envs, gamma=config[Environment.GAMMA])
        else:
            envs = VecNormalize(envs, ret=False)

    envs = VecPyTorch(envs, device)

    if num_frame_stack or len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs
