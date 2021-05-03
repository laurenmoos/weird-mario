import argparse
import toml
from enum import Enum

import pathlib

CONFIG_DIRECTORY = "/configs/"


class Environment:
    NAME = 'name'
    SEED = 'seed'
    NUM_PROCESSES = 'num_processes'
    GAMMA = 'gamma'
    USE_WEIRD = 'use_weird'
    WITH_AUTOSHROOM = 'with_autoshroom'
    LEVEL = 'level'
    VALIDATION_LEVEL = 'validation_level'
    ALLOW_EARLY_RESETS = 'allow_early_resets'
    EPISODE_LENGTH = 'episode_length'
    USE_SKIP = 'use_skip'
    REWARD = 'reward'
    REWARD_TRACE = 'reward_trace'
    ENV_STEPS = 'env_steps'
    USE_PROPER_TIME_LIMITS = 'use_proper_time_limits'


class Logging:
    LOG_INTERVAL = 'log_interval'
    LOG_DIR = 'log_dir'
    EVAL_LOG_DIR = 'eval_log_dir'


class Agent:
    TRACE_SIZE = 'trace_size'
    EVAL_INTERVAL = 'eval_interval'
    SAVE_INTERVAL = 'save_interval'
    SAVE_DIR = 'save_dir'
    ALGO = "algo"
    NUM_UPDATES = 'num_updates'
    NUM_STEPS = 'num_steps'
    USE_ACKTR = 'use_acktr'
    # a2c
    VALUE_LOSS_COEFF = 'value_loss_coeff'
    ENTROPY_COEF = 'entropy_coef'
    MAX_GRAD_NORM = 'max_grad_norm'
    ALPHA = 'alpha'
    LEARNING_RATE = 'learning_rate'
    # ppo
    IS_HEADLESS = 'is_headless'
    CLIP = 'clip'
    EPOCH = 'epoch'
    NUM_MINI_BATCH = 'num_mini_batch'
    USE_CLIPPED_VALUE_LOSS = 'use_clipped_value_loss'
    USE_LINEAR_LR_DECAY = 'use_linear_lr_decay'


class PolicyConfig:
    BASE_NETWORK_TYPE = 'base_network_type'
    BASE_NETWORK_TOPOLOGY = 'base_network_topology'
    USE_RECURRENT_POLICY = 'use_recurrent_policy'


class Config:
    _instance = None

    def __init__(self):
        raise RuntimeError

    def environment(self) -> dict:
        return self.environment

    def policy(self) -> dict:
        return self.policy

    def logging(self) -> dict:
        return self.logging

    def device(self) -> str:
        return self.device

    def load(self) -> str:
        return self.load

    def agent(self) -> dict:
        return self.agent

    @classmethod
    def instance(cls, args=None):
        if cls._instance is None:
            cls._instance = Config.__new__(cls)
            cls.config = parse_config(args)
            cls.environment = cls.config.environment
            cls.policy = cls.config.policy
            cls.logging = cls.config.logging
            cls.agent = cls.config.agent
            cls.device = cls.config.device
            cls.load = cls.config.load
        return cls._instance


def parse_config(args):
    config = toml.load(str(pathlib.Path().absolute()) + CONFIG_DIRECTORY + args.exp_name + '.toml')
    for k in config.keys():
        args.__dict__[k.replace('-', '_')] = config[k]
    return args


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--exp-name',
        help='Name for experiment used to retrieve cofngiuration file'
    )
    parser.add_argument(
        '--load',
        help='Load pre-trained model',
        default=False
    )
    parser.add_argument(
        '--device',
        help='Device used to run experiments {cpu, gpu}'
    )

    parser.add_argument(
        '--config',
        default='',
        help='toml file containing configuration for run'
    )
    args = parser.parse_args()

    return args
