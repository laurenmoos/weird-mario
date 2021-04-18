import logging

from utils.train_utils import update_linear_schedule

'''
Base agent for policy gradient models.
'''


class PolicyGradient:

    def __init__(self, config):
        self.logger = logging.getLogger("Agent")

    #TODO; custom override
    def train(self, config, optimizer):
        """
        Main training loop
        :return:
        """
        for j in range(config.num_updates):

            if config.use_linear_lr_decay:
                # decrease learning rate linearly
                update_linear_schedule(optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):
                train_one_epoch
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.tobs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

    def train_one_epoch(self):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.tobs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        tobs = []
        if not args.headless:
            envs.render()
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_crash.append(info['crash'])
                if info['crash'] == 1:
                    crash_rewards.append(info['episode']['r'])

            trace = info['trace'][0:trace_size]
            trace = [x[2] for x in trace]
            word_to_ix = toke.tokenize(trace)
            seq = prepare_sequence(trace, word_to_ix)
            if len(seq) < trace_size:
                seq = torch.zeros((trace_size), dtype=torch.long)
            seq = seq[:trace_size]
            # print (seq.dtype)
            tobs.append(seq)
        tobs = torch.stack(tobs)
        # print (tobs)
        # print (tobs.size())
        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])
        rollouts.insert(obs, tobs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError
