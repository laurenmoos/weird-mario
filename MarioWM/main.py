#! /usr/bin/env python
import torch
import datetime
import logging

from a2c_ppo_acktr.agents.policy_gradient import a2c_acktr, policy_gradient
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.graphs.models.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from a2c_ppo_acktr.utils.config import parse_config
from a2c_ppo_acktr.utils.system_utils import cleanup_log_dir
from a2c_ppo_acktr.arguments import get_args

import os

logdir = 'runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# default `log_dir` is "runs" - we'll be more specific here

# vanilla logger
logger = logging.getLogger("Agent")

# TODO: replace
args = get_args()


def main():
    config = parse_config(args)

    env, policy, logging, agent = config.environment, config.policy, config.logging, config.agent

    print(logging.keys())
    '''
    Setting up Environment
    '''
    # default to hard-coded log dir if not present in config
    if logging['log_dir']:
        logdir = logging['log_dir']

    logger.info("Logging to {}".format(logdir))
    log_dir = os.path.expanduser(logdir)
    eval_log_dir = logdir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    logger.info("Spinning up {} thread(s)".format(1))
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.device == 'gpu' else "cpu")

    envs = make_vec_envs(log_dir, device, policy['use_skip'], env)

    # initialize actor-critic policy
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': policy['use_recurrent_policy']})
    actor_critic.to(device)

    '''
    Initialize Agent and maybe retrieve pre-trained network.
    '''
    save_path = os.path.join(config.save_dir, config.algo)

    if config.load:
        logger.info("Retrieving pre-trained network at: {)".format(save_path))
        actor_critic.load_state_dict = (os.path.join(save_path, config.env_name + ".pt"))

    logger.log("Spinning up agent: {)".format(config.algo))
    if config.algo == 'a2c':
        agent = agents.A2C_ACKTR(actor_critic, config)
    elif config.algo == 'ppo':
        agent = agents.PPO(actor_critic, config)
    elif config.algo == 'acktr':
        agent = agents.A2C_ACKTR(actor_critic, config)

    '''
    Create Rollout Storage Object 
    '''
    rollouts = RolloutStorage(config.param.num_steps, config.param.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    '''
    Reset Observations 
    '''
    obs = envs.reset()
    tobs = torch.zeros((config.System.num_processes, config.logging.trace_size), dtype=torch.long)
    rollouts.obs[0].copy_(obs)
    rollouts.tobs[0].copy_(tobs)

    rollouts.to(device)

    '''
    Train
    '''
    agent.train(config, rollouts)


if __name__ == "__main__":
    main()
