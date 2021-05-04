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
import mlflow
from mlflow import log_metric, log_metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('PolicyGradient')
logdir = 'runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

writer = SummaryWriter(logdir)
from arguments import Config, Agent, Environment, Logging


def get_config():
    return Config.instance()


'''
Base agent for policy gradient models.
'''


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


def _validate(actor_critic, ob_rms, device):
    config = get_config()
    env_config = config.environment

    eval_envs = make_vec_envs(device)
    vec_norm = get_vec_normalize(eval_envs)

    if vec_norm:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        env_config[Environment.NUM_PROCESSES], actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(env_config[Environment.NUM_PROCESSES], 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                masks=eval_masks,
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

    log.info(" Evaluation using {} episodes: mean reward {:.5f}"
             .format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))


def _checkpoint(actor_critic, envs, toke, env_name):
    config = get_config()
    agent_config = config.agent

    save_path = os.path.join(agent_config[Agent.SAVE_DIR], agent_config[Agent.ALGO])
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    torch.save([actor_critic, getattr(get_vec_normalize(envs), 'ob_rms', None)],
               os.path.join(save_path, env_name + ".pt"))
    pickle.dump(toke.word_to_ix, open("save.p", "wb"))


def _log(start, total_num_steps, episode_rewards, episode_crash, crash_rewards):
    end = time.time()
    with mlflow.start_run():
        # TODO: add dist entropy
        log_metric(key='fps', value=int(total_num_steps / (end - start)))
        log_metrics(episode_rewards)
        log_metrics(episode_crash)
        log_metrics(crash_rewards)


class PolicyGradient:

    def __init__(self, actor_critic, config):
        self.config = config
        self.actor_critic = actor_critic
        self.logger = logging.getLogger("Agent")

    def update(self, rollouts):
        raise NotImplementedError

    def update_linear_schedule(self, j, num_updates, lr):
        raise NotImplementedError

    def train(self, actor_critic, rollouts, envs, device):
        config = get_config()
        agent_config, env_config, logging_config = config.agent, config.environment, config.logging

        num_processes, env_steps, agent_steps = env_config[Environment.NUM_PROCESSES], \
                                                env_config[Environment.ENV_STEPS], \
                                                agent_config[Agent.NUM_STEPS]

        episode_rewards = deque(maxlen=num_processes * 3)
        episode_crash = deque(maxlen=num_processes * 3)
        crash_rewards = deque(maxlen=num_processes * 3)

        start = time.time()
        num_updates = int(env_steps) // agent_steps // num_processes

        log.info("Running training loop with {} updates".format(num_updates))

        toke = tokenizer()

        '''Environment Step'''
        for j in range(num_updates):

            if agent_config[Agent.USE_LINEAR_LR_DECAY]:
                # decrease learning rate linearly
                self.update_linear_schedule(j, num_updates, agent_config[Agent.LEARNING_RATE])

            '''Agent Step'''
            for step in range(agent_steps):
                self.train_one_epoch(step, rollouts, envs, episode_rewards, episode_crash, crash_rewards, toke)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1], rollouts.tobs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, True, env_config[Environment.GAMMA], 0.95,
                                     env_config[Environment.USE_PROPER_TIME_LIMITS])

            value_loss, action_loss, dist_entropy = self.update(rollouts)

            with mlflow.start_run():
                log_metric(key='value_loss', value=value_loss)
                log_metric(key='action_loss', value=action_loss)
                log_metric(key='dist_entropy', value=dist_entropy)

            rollouts.after_update()

            # checkpoint periodically
            if j % agent_config[Agent.SAVE_INTERVAL] == 0 or j == num_updates - 1 and agent_config[Agent.SAVE_DIR]:
                _checkpoint(actor_critic, envs, toke, env_config[Environment.NAME])

            if j % agent_config[Agent.EVAL_INTERVAL] == 0:
                normalized = get_vec_normalize(envs)
                ob_rms = None
                if normalized:
                    ob_rms = normalized.ob_rms
                _validate(actor_critic, ob_rms, device)

            # At configured log interval step, log rewards and reward metadata from the episode
            if j % logging_config[Logging.LOG_INTERVAL] == 0:
                log.info("Meta-Logging")
                total_num_steps = (j + 1) * num_processes * agent_steps
                _log(start, total_num_steps, episode_rewards, episode_crash, crash_rewards)

    def train_one_epoch(self, step, rollouts, envs, episode_rewards, episode_crash, crash_rewards, toke):
        agent_config = get_config().agent

        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                rollouts.obs[step], rollouts.tobs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        obs, reward, done, infos = envs.step(action)
        tobs = []
        if not agent_config[Agent.IS_HEADLESS]:
            envs.render()
        for info in infos:
            if 'episode' in info.keys():
                log.info("episode is present in info keys")
                episode_rewards.append(info['episode']['r'])
                episode_crash.append(info['crash'])
                if info['crash'] == 1:
                    crash_rewards.append(info['episode']['r'])

            trace_size = agent_config[Agent.TRACE_SIZE]
            trace = info['trace'][0:trace_size]
            trace = [x[2] for x in trace]
            word_to_ix = toke.tokenize(trace)
            seq = prepare_sequence(trace, word_to_ix)
            if len(seq) < trace_size:
                seq = torch.zeros(trace_size, dtype=torch.long)
            seq = seq[:trace_size]
            tobs.append(seq)
        tobs = torch.stack(tobs)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        rollouts.insert(obs, tobs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)
