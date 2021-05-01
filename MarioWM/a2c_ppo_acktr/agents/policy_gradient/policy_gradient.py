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

writer = SummaryWriter(logdir)

class tokenizer:

    def __init__(self):
        self.word_to_ix = {}

    def tokenize(self, line):
        self.word_to_ix = {}

        for word in line:
            if word not in self.word_to_ix:  # word has not been assigned an index yet
                # print (len (self.word_to_ix))
                self.word_to_ix[word] = len(self.word_to_ix)  # Assign each word with a unique index
        return self.word_to_ix


def prepare_sequence(line, to_ix):
    idxs = []
    for word in line:
        idxs.append(to_ix[word])
    return torch.tensor(idxs, dtype=torch.long)

def _validate(log_dir, device, config, env, actor_critic, envs, episode_rewards):
    log.info("Validating....")

    ob_rms = get_vec_normalize(envs).ob_rms



    eval_envs = make_vec_envs(env, log_dir, device)

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
        obs, _, done, infos = eval_envs.step(action)

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

def _checkpoint(agent, actor_critic, envs, toke, env_name):

        save_path = os.path.join(agent['save_dir'], agent['algo'])
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        torch.save([
            actor_critic,
            getattr(get_vec_normalize(envs), 'ob_rms', None)
        ], os.path.join(save_path, env_name + ".pt"))
        pickle.dump(toke.word_to_ix, open("save.p", "wb"))

def _log(total_num_steps, episode_rewards, dist_entropy, value_Loss, action_loss, episode_crash, crash_rewards):
    end = time.time()

    fps = int(total_num_steps / (end - start))
    log.info("Updates {}, timesteps {}, FPS {} ".format(j, total_num_steps, fps))

    log.info("Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
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



class PolicyGradient:

    def __init__(self, actor_critic, config):
        self.config = config
        self.actor_critic = actor_critic
        self.logger = logging.getLogger("Agent")

    def update(self, rollouts):
        raise NotImplementedError

    def train(self, agent, env, logging, actor_critic, rollouts, envs, log_dir, device):

        num_processes, env_steps, agent_steps = env['num_processes'], env['env_steps'], agent['num_steps']

        episode_rewards = deque(maxlen=num_processes * 3)
        episode_crash = deque(maxlen=num_processes * 3)
        crash_rewards = deque(maxlen=num_processes * 3)

        start = time.time()
        num_updates = int(env_steps) // agent_steps // num_processes

        log.info("Running training loop with {} updates".format(num_updates))

        toke = tokenizer()

        for j in range(num_updates):

            if agent['use_linear_lr_decay']:
                # decrease learning rate linearly
                # update_linear_schedule(optimizer, j, num_updates, agent.optimizer.lr
                # if args.algo == "acktr" else args.lr)
                lr = self.optimizer.lr if self.acktr else agent['learning_rate']
                update_linear_schedule(self.optimizer, j, num_updates, lr)

            for step in range(agent_steps):
                self.train_one_epoch(step, rollouts, envs, episode_rewards, episode_crash, crash_rewards,
                                     agent, toke)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1], rollouts.tobs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, False, env['gamma'], None, env['use_proper_time_limits'])

            value_loss, action_loss, dist_entropy = self.update(rollouts)

            rollouts.after_update()

            # checkpoint periodically
            if (j % agent['save_interval'] == 0 or j == num_updates - 1) and agent['save_dir'] != "":
                _checkpoint(agent, actor_critic, envs, toke, env['name'])

            if agent['eval_interval'] is not None and len(episode_rewards) > 1 and j % agent['eval_interval'] == 0:
                _validate(log_dir, device, agent, env, actor_critic, envs, episode_rewards )

            # At configured log interval step, log rewards and reward metadata from the episode
            if j % logging['log_interval'] == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * num_processes * agent_steps
                _log(total_num_steps, episode_rewards, dist_entropy, value_Loss, action_loss, episode_crash, crash_rewards)


    def train_one_epoch(self, step, rollouts, envs, episode_rewards, episode_crash, crash_rewards, agent, toke):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                rollouts.obs[step], rollouts.tobs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        tobs = []
        if not agent['is_headless']:
            envs.render()
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_crash.append(info['crash'])
                if info['crash'] == 1:
                    crash_rewards.append(info['episode']['r'])

            trace_size = agent['trace_size']
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

