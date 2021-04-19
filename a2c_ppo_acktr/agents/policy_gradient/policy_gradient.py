import torch

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from ...envs import make_vec_envs
from ...utils.system_utils import get_vec_normalize

from collections import deque
import os, time
from ...utils.train_utils import update_linear_schedule
import datetime

import pickle


import logging

log = logging.getLogger('PolicyGradient')
logdir = 'runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

writer = SummaryWriter(logdir)


'''
Base agent for policy gradient models.
'''


def _validate(config, actor_critic, envs, episode_rewards):

    ob_rms = get_vec_normalize(envs).ob_rms

    eval_envs = make_vec_envs(config.env_name, config.seed + config.num_processes, config.num_processes,
                              None, config.eval_log_dir, config.device, True)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        config.num_processes, actor_critic.recurrent_hidden_state_size, device=config.device)
    eval_masks = torch.zeros(config.num_processes, 1, device=config.device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.long,
            device=config.device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n"
          .format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))


class PolicyGradient:

    def __init__(self, actor_critic, config):
        self.config = config
        self.actor_critic = actor_critic
        self.logger = logging.getLogger("Agent")

    def update(self, rollouts):
        raise NotImplementedError

    def train(self, config, actor_critic, optimizer, rollouts, envs):

        for j in range(config.num_updates):

            if config.use_linear_lr_decay:
                # decrease learning rate linearly
                # update_linear_schedule(optimizer, j, num_updates, agent.optimizer.lr
                # if args.algo == "acktr" else args.lr)
                lr = self.optimizer.lr if self.acktr else config.lr
                update_linear_schedule(self.optimizer, j, config.num_updates, lr)

            for step in range(config.num_steps):
                self.train_one_epoch(step, rollouts, envs, episode_rewards, episode_crash, crash_rewards)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1], rollouts.tobs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, config.use_gae, config.gamma, config.gae_lambda,
                                     config.use_proper_time_limits)

            self.update(rollouts)

            rollouts.after_update()

            system, param = config.system, config.param

            episode_rewards = deque(maxlen=system.num_processes * 3)
            episode_crash = deque(maxlen=system.num_processes * 3)
            crash_rewards = deque(maxlen=system.num_processes * 3)

            start = time.time()
            num_updates = int(param.num_env_steps) // param.num_steps // param.num_processes

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.tobs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, param.use_gae, param.gamma, param.gae_lambda, param.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.update(rollouts)

            rollouts.after_update()

            check = config.checkpoint

            # checkpoint periodically
            if (j % check.save_interval == 0 or j == num_updates - 1) and check.save_dir != "":
                save_path = os.path.join(check.save_dir, param.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, config.env_name + ".pt"))
                pickle.dump(toke.word_to_ix, open("save.p", "wb"))

            # At configured log interval step, log rewards and reward metadata from the episode
            if j % config.log.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * param.num_processes * param.num_steps

            if config.eval_interval is not None and len(episode_rewards) > 1 and j % config.eval_interval == 0:
                _validate(config, actor_critic, envs, episode_rewards)

            end = time.time()

            fps = int(total_num_steps / (end - start))
            log.log("Updates {}, timesteps {}, FPS {} ".format(j, total_num_steps, fps))

            log.log("Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                    .format(len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

            writer.add_scalar('mean reward', np.mean(episode_rewards), total_num_steps, )
            writer.add_scalar('std', np.std(episode_rewards), total_num_steps, )
            writer.add_scalar('median reward', np.median(episode_rewards), total_num_steps, )
            writer.add_scalar('max reward', np.max(episode_rewards), total_num_steps, )
            writer.add_scalar('crash frequency', np.mean(episode_crash), total_num_steps, )
            writer.add_scalar('median crash reward', np.median(crash_rewards), total_num_steps, )
            writer.add_scalar('mean crash reward', np.mean(crash_rewards), total_num_steps, )

    def train_one_epoch(self, step, rollouts, envs, episode_rewards, episode_crash, crash_rewards):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                rollouts.obs[step], rollouts.tobs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step()
        tobs = []
        if not self.config.args.headless:
            envs.render()
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_crash.append(info['crash'])
                if info['crash'] == 1:
                    crash_rewards.append(info['episode']['r'])

            trace_size = self.config.logging.trace_size
            trace = info['trace'][0:trace_size]
            trace = [x[2] for x in trace]
            word_to_ix = toke.tokenize(trace)
            seq = prepare_sequence(trace, word_to_ix)
            if len(seq) < trace_size:
                seq = torch.zeros((trace_size), dtype=torch.long)
            seq = seq[:trace_size]
            tobs.append(seq)
        tobs = torch.stack(tobs)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])
        rollouts.insert(obs, tobs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)

