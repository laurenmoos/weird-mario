#! /usr/bin/env python
import torch
import logging

from a2c_ppo_acktr.agents.policy_gradient import a2c_acktr, ppo
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.graphs.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from arguments import get_args, Config, Environment, Agent, PolicyConfig, Logging
from mlflow import mlflow, log_params
import os

# default `log_dir` is "runs" - we'll be more specific here

# vanilla logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")


def main():
    # initialize config
    args = get_args()
    config = Config.instance(args)

    env, policy, logging, agent_config = config.environment, config.policy, config.logging, config.agent

    # log params

    with mlflow.start_run():
        log_params(env)
        log_params(policy)
        log_params(agent_config)

    '''
    Setting up Environment
    '''

    torch.manual_seed(env[Environment.SEED])
    torch.cuda.manual_seed_all(env[Environment.SEED])

    if config.device == 'gpu' and torch.cuda.is_available() and True:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.device == 'gpu' else "cpu")

    logger.info("Setting up environment {} on device {}".format(env, device))
    envs = make_vec_envs(device)

    observation_space_shape, action_space = envs.observation_space.shape, envs.action_space

    logger.info("{} dim observation space".format(observation_space_shape))

    # initialize actor-critic policy
    logger.info("Initializing policy {}".format(policy))
    actor_critic = Policy(observation_space_shape, action_space,
                          base_kwargs={'recurrent': policy[PolicyConfig.USE_RECURRENT_POLICY]})
    actor_critic.to(device)

    '''
    Initialize Agent and maybe retrieve pre-trained network.
    '''
    save_path = os.path.join(agent_config[Agent.SAVE_DIR], agent_config[Agent.ALGO])

    if config.load:
        logger.info("Retrieving pre-trained network at: {)".format(save_path))
        actor_critic.load_state_dict = (os.path.join(save_path, env[Environment.NAME] + ".pt"))

    logger.info("Spinning up agent: {}".format(agent_config))
    agent_x = None
    if agent_config[Agent.ALGO] == 'a2c':
        agent_x = a2c_acktr.A2C_ACKTR(actor_critic, False)
    elif agent_config[Agent.ALGO] == 'ppo':
        agent_x = ppo.PPO(actor_critic)
    elif agent_config[Agent.ALGO] == 'acktr':
        agent_x = a2c_acktr.A2C_ACKTR(actor_critic, False)

    '''
    Create Rollout Storage Object - this vectorizes the previously configured environment in 
    a way bound to the configured agent 
    '''
    logger.info("Vectorizing environment {} for agent {}".format(env, agent_x))
    rollouts = RolloutStorage(agent_config[Agent.NUM_STEPS], env[Environment.NUM_PROCESSES],
                              observation_space_shape, action_space,
                              actor_critic.recurrent_hidden_state_size, agent_config[Agent.TRACE_SIZE])
    '''
    Reset Observations 
    '''
    # TODO: this could be moved to inside rollout storage
    logger.info("Resetting environment {}".format(env))
    obs = envs.reset()
    tobs = torch.zeros((env[Environment.NUM_PROCESSES], agent_config[Agent.TRACE_SIZE]), dtype=torch.long)
    rollouts.obs[0].copy_(obs)
    rollouts.tobs[0].copy_(tobs)

    rollouts.to(device)

    '''
    Train
    '''
    logger.info("Training agent {} with environment {}".format(agent_config, env))
    agent_x.train(actor_critic, rollouts, envs)


if __name__ == "__main__":
    main()
