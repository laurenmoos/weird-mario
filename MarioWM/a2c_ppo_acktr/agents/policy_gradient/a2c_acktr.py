import torch
import torch.nn as nn
import torch.optim as optim

from ..optim.kfac import KFACOptimizer

from .policy_gradient import PolicyGradient
import numpy as np
from ...utils.system_utils import get_vec_normalize
from ...envs import make_vec_envs



class A2C_ACKTR(PolicyGradient):
    def __init__(self, actor_critic, config, is_acktr):
        super().__init__()

        self.actor_critic = actor_critic

        self.value_loss_coef = config['value_loss_coef']
        self.entropy_coef = config['entropy_coef']

        self.max_grad_norm = config['max_grad_norm']

        self.alpha = config['alpha']
        self.lr = config['learning_rate']
        #TODO: should standardize this
        self.eps = 1.e-8

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), self.lr, eps=self.eps, alpha=self.alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see https://arxiv.org/abs/1412.1193
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

        if self.acktr is False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def eval(self, ob_rms, system):
        eval_envs = make_vec_envs(system.env_name, system.seed + system.num_processes, system.num_processes,
                                  None, system.eval_log_dir, system.device, True)

        vec_norm = get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms

        eval_episode_rewards = []

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(system.num_processes,
                                                   self.actor_critic.recurrent_hidden_state_size, device=system.device)
        eval_masks = torch.zeros(system.num_processes, 1, device=system.device)

        while len(eval_episode_rewards) < 10:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step()

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.long,
                device=system.device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        eval_envs.close()

        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))
