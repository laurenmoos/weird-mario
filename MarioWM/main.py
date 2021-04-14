#! /usr/bin/env python

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import datetime
import pickle

from torch.utils.tensorboard import SummaryWriter

import os


logdir = 'runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(logdir)



class tokenizer:

    def __init__(self,args):
        if args.load:         
            self.word_to_ix = pickle.load( open( "save.p", "rb" ) )             
        else:
            self.word_to_ix = {}
        
    def tokenize (self, line):
  
        for word in line:
            if word not in self.word_to_ix:  # word has not been assigned an index yet
                #print (len (self.word_to_ix))
                self.word_to_ix[word] = len(self.word_to_ix)  # Assign each word with a unique index
        return self.word_to_ix
        
   

def prepare_sequence(line, to_ix):

    idxs=[]
    for word in line:
        idxs.append (to_ix[word])
    return torch.tensor(idxs, dtype=torch.long)

def main():
    args = get_args()
    trace_size = args.trace_size    
    toke = tokenizer(args)

    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)

    #if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        #torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    
    save_path = os.path.join(args.save_dir, args.algo)
    
    if args.load:
        actor_critic.load_state_dict = (os.path.join(save_path, args.env_name + ".pt"))
        
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)



    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
 
    obs = envs.reset()
    tobs = torch.zeros((args.num_processes,trace_size), dtype=torch.long)
    #print (tobs.dtype)
    rollouts.obs[0].copy_(obs)
    rollouts.tobs[0].copy_(tobs)

    rollouts.to(device)

    episode_rewards = deque(maxlen=args.num_processes*3)
    episode_crash = deque(maxlen=args.num_processes*3)
    crash_rewards = deque(maxlen=args.num_processes*3)
    
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
  
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            
            
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],rollouts.tobs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            tobs = []
            if not args.headless:
                envs.render()
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_crash.append(info ['crash'])
                    if info['crash'] == 1:
                        crash_rewards.append(info['episode']['r'])
                    
                    
                trace = info['trace'] [0:trace_size]
                trace = [x[2] for x in trace]
                word_to_ix = toke.tokenize(trace)               
                seq = prepare_sequence(trace, word_to_ix)
                if len (seq)< trace_size:
                    seq = torch.zeros((trace_size), dtype=torch.long)
                seq = seq[:trace_size]
                #print (seq.dtype)
                tobs.append (seq)
            tobs = torch.stack(tobs)  
            #print (tobs)
            #print (tobs.size())
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, tobs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.tobs[-1],rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        #"""
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))
            pickle.dump (toke.word_to_ix, open( "save.p", "wb" ) )

        #"""
        
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            writer.add_scalar('mean reward', np.mean(episode_rewards) ,total_num_steps, )
            writer.add_scalar('std', np.std(episode_rewards) ,total_num_steps, )
            writer.add_scalar('median reward', np.median (episode_rewards), total_num_steps,)
            writer.add_scalar('max reward', np.max(episode_rewards), total_num_steps,)
            writer.add_scalar('crash frequency', np.mean (episode_crash), total_num_steps,)
            writer.add_scalar('median crash reward', np.median(crash_rewards), total_num_steps,)
            writer.add_scalar('mean crash reward', np.mean(crash_rewards), total_num_steps,)


            

            


        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()