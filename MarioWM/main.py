#! /usr/bin/env python
import torch

from a2c_ppo_acktr import agents, utils
from a2c_ppo_acktr.envs import make_vec_envs
from MarioWM.a2c_ppo_acktr.graphs.models.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
import datetime
import logging

from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr.utils.system_utils import cleanup_log_dir

import os

logdir = 'runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(logdir)

#vanilla logger
logger = logging.getLogger("Agent")


# TODO: pytorch lightning
def main():
    #TODO make this real
    config = get_config_from_json(args.exp_name)

    #TODO: Log the json file picked up for the experiment here
    logger.log("Running experiment {} using Configuration file: {}".format(args.exp_name, config))

    logger.log("Logging to {}".format(logdir))
    log_dir = os.path.expanduser(logdir)
    eval_log_dir = logdir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    logger.log("Running with {} thread(s)".format(1))
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(config.environment.name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

    # initialize actor-critic policy
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': config.recurrent_policy})
    actor_critic.to(device)

    save_path = os.path.join(args.save_dir, config.algo)

    # maybe load pre-trained network
    if args.load:
        actor_critic.load_state_dict = (os.path.join(save_path, config.env_name + ".pt"))

    logger.log("Applying policy to agent: {)".format(config.algo))
    if config.algo == 'a2c':
        agent = agents.A2C_ACKTR(actor_critic, config)
    elif config.algo == 'ppo':
        agent = agents.PPO(config)
    elif config.algo == 'acktr':
        agent = agents.A2C_ACKTR(actor_critic, config)

    '''
    Create rollout storage object 
    '''
    rollouts = RolloutStorage(config.param.num_steps, config.param.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    #TODO: change this to an "initialize observations" function
    obs = envs.reset()
    tobs = torch.zeros((config.System.num_processes, config.logging.trace_size), dtype=torch.long)
    # print (tobs.dtype)
    rollouts.obs[0].copy_(obs)
    rollouts.tobs[0].copy_(tobs)

    rollouts.to(device)


    agent.train()
    agent.eval()





if __name__ == "__main__":
    main()
