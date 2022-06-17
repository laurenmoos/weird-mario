#! /usr/bin/env python
import torch
import logging

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.storage import RolloutStorage

import pytorch_lightning as pl
from arguments import get_args

logger = logging.getLogger("Agent")

if __name__ == "__main__":
    # parse args
    parser = get_args()
    torch.set_num_threads(1)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")

    logger.info(f"Setting up environment, logging to {args.logdir}")

    # construct environment
    envs = make_vec_envs(log_dir, device, env)

    logger.info(
        f"Intiialized environment has {envs.observation_space.shape} dim observation space and {envs.action_space.shape} dim action space")

    # TODO: does proximal policy optimization require a rollout storage
    logger.log("Vectorizing environment {} for agent {}".format(env, agent))
    rollouts = RolloutStorage(agent['num_steps'], env['num_processes'],
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    tobs = torch.zeros((env['num_processes'], agent['trace_size']), dtype=torch.long)
    rollouts.obs[0].copy_(obs)
    rollouts.tobs[0].copy_(tobs)

    rollouts.to(device)

    # TODO: have a switch statement here to run model or a benchmark
    model = ProximaLPolicyOptimization(config)
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=1000
    )
    parser = trainer.add_argparse_args(parser)

    trainer.fit(model)
