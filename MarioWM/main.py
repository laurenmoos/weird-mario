#! /usr/bin/env python
import torch
import datetime
import logging

from a2c_ppo_acktr.agents.policy_gradient import a2c_acktr, ppo
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")

args = get_args()


def main():
    config = parse_config(args)

    env, policy, logging, agent = config.environment, config.policy, config.logging, config.agent

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

    torch.manual_seed(env['seed'])
    torch.cuda.manual_seed_all(env['seed'])

    if config.device == 'gpu' and torch.cuda.is_available() and True:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    logger.info("Spinning up {} thread(s)".format(1))
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.device == 'gpu' else "cpu")

    logger.info("Setting up environment {} on device {} logging to {}".format(env, device, log_dir))
    envs = make_vec_envs(env, log_dir, device)

    logger.info("Intiialized environment has {} dim observation space and {} dim action space "
                .format(envs.observation_space.shape, envs.action_space.shape))


    # initialize actor-critic policy
    logger.info("Initializing policy {}".format(policy))
    actor_critic = Policy(policy, envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': policy['use_recurrent_policy']})
    actor_critic.to(device)

    '''
    Initialize Agent and maybe retrieve pre-trained network.
    '''
    save_path = os.path.join(agent['save_dir'], agent['algo'])

    if config.load:
        logger.info("Retrieving pre-trained network at: {)".format(save_path))
        actor_critic.load_state_dict = (os.path.join(save_path, env['name'] + ".pt"))

    logger.info("Spinning up agent: {}".format(agent))
    if agent['algo'] == 'a2c':
        agent_x = a2c_acktr.A2C_ACKTR(actor_critic, agent, False)
    elif agent['algo'] == 'ppo':
        agent_x = ppo.PPO(actor_critic, agent)
    elif agent['algo'] == 'acktr':
        agent_x = a2c_acktr.A2C_ACKTR(actor_critic, agent, False)

    '''
    Create Rollout Storage Object - this vectorizes the previously configured environment in 
    a way bound to the configured agent 
    '''
    #TODO: i believe this is a2c speciifc?
    logger.info("Vectorizing environment {} for agent {}".format(env, agent_x))
    rollouts = RolloutStorage(agent['num_steps'], env['num_processes'],
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, agent['trace_size'])
    '''
    Reset Observations 
    '''
    #TODO: this could be moved to inside rollout storage
    logger.info("Resetting environment {}".format(env))
    obs = envs.reset()
    tobs = torch.zeros((env['num_processes'], agent['trace_size']), dtype=torch.long)
    rollouts.obs[0].copy_(obs)
    rollouts.tobs[0].copy_(tobs)

    rollouts.to(device)

    '''
    Train
    '''
    logger.info("Training agent {} with environment {}".format(agent, env))
    agent_x.train(agent, env, logging, actor_critic, rollouts, envs, log_dir, device)


if __name__ == "__main__":
    main()
