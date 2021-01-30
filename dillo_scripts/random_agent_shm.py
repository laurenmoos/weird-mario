#script for experimenting with dumping out gym info and other states when Mario dies or other things happen
import retro
import pandas as pd
import numpy as np

import argparse
import signal
import time
import os
import struct
import sys
import sysv_ipc as ipc
import random

parser = argparse.ArgumentParser()
parser.add_argument('--game', default='SuperMarioWorld-Snes', help='the name or path for the game to run')
parser.add_argument('--state', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
parser.add_argument('--shmkey', '-k', type=int, default=1337, help='shm key for register diagnostics')
args = parser.parse_args()

obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)
verbosity = args.verbose - args.quiet

visited = set()

#setup our history buffers

pc_hist=np.empty(100,)
obshist=np.empty(141312,)

#Attach to shared memory

stok = int(args.shmkey)
key=ipc.ftok('/dev/shm', stok)
shm = ipc.SharedMemory(key,0,0) 
shm.attach()

try:
    while True:
        ob = env.reset()
        t = 0
        totrew = [0] * args.players
        last_moves=[]
        last_obs=[]
        deadcount=0       
        pc_hist=[]
         
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)
            #print(info)
            regbuf=shm.read(4) 
            pc = struct.unpack("<I", regbuf)[0]
            visited.add(pc)
            t += 1
            if info['dead'] > 0:
               last_moves.append(info)
               last_obs.append(ob)
               pc_hist.append(pc)
               deadcount=0
            else:
               deadcount +=1
               print(len(last_moves))
 
            if deadcount == 1:
               print("***********PLAYER DIED: INFO DUMP*************")    
             
               with open('./dead_drop','a') as dd:
                  for n in range(0,len(last_moves)):
                     infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in last_moves[n].items()])
                     print(infostr)
                     dumpstr="t="+str(t)+" "+infostr+" PC:"+str(pc_hist[n])
                     dd.write(dumpstr)
                     dd.write('\n')

               print("ob type=%s ob_len=%d ob_dim %s" % (type(ob),len(ob),ob.ndim))
               print(len(last_obs[0]))
               print(len(last_obs))
               print(last_obs[0])
               print(type(last_obs[0]))
               print(np.shape(last_obs))

             #  df = pd.DataFrame(ob,columns=['Mem value at'])
             #  df.to_csv(r'ram_snapshot.csv')
             #  print("***********PLAYER DIED: INFO DUMP*************")       
             #  dd.close()
             #  #do.close()
               
            if t % 10 == 0:
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                env.render()

            #Every hundred actions, flush the move history and PC counter history 
            if t % 100 == 0:
               last_moves=[]
               pc_hist=[]
           
            if args.players == 1:
                rew = [rew]
            for i, r in enumerate(rew):
                totrew[i] += r
                if verbosity > 0:
                    if r > 0:
                        print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                        with open('./reward_drop','a') as rd:
                           for n in range(0,len(last_moves)):
                              infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in last_moves[n].items()])
                             # dumpstr="t="+str(t)+" "+infostr
                              dumpstr="t="+str(t)+" "+infostr+" PC:"+str(pc_hist[n])
                              rd.write(dumpstr)
                              rd.write('\n')
                           rd.close()
 
                    if r < 0:
                        print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
            if done:
                env.render()
                try:
                    if verbosity >= 0:
                        if args.players > 1:
                            print("done! total reward: time=%i, reward=%r" % (t, totrew))
                        else:
                            print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                break
except KeyboardInterrupt:
    shm.detach()
    exit(0)
