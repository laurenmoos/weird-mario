#! /usr/bin/env python

import signal
import time
import os
import struct
import sys
import retro
import sysv_ipc as ipc
import random


DELAY = 0
CONTINUE = True
VISITED = set()


def signal_handler(sig, frame):
    global CONTINUE
    CONTINUE = False
    print(f"SIGINT Received.\n{len(VISITED)} Addresses Visited:")
    for addr in sorted(VISITED):
        print(f"    0x{addr:x}")
    sys.exit(0)


def mem_blocks(env):
    return [range(p[0], p[0]+len(p[1])) for p in env.data.memory.blocks.items()]


def random_address(env):
    blocks = mem_blocks(env)
    block = random.choice(blocks)
    return random.choice(block)


def random_poke(env):
    val = random.randint(0, 256)
    addr = random_address(env)
    return retro._retro.Memory.assign(env.data.memory, addr, "uint8", val)


def get_key():
    retro_run_id = os.getenv("RETRO_RUN_ID")
    if retro_run_id is None:
        print("RETRO_RUN_ID not set")
        sys.exit(1)
    rid = int(retro_run_id)
    return ipc.ftok("/dev/shm", rid)


def attach_shm():
    key = get_key()
    shm = ipc.SharedMemory(key, 0, 0)
    shm.attach()
    return shm


def read_pc(shm):
    buf = shm.read(4)
    pc = struct.unpack("<I", buf)[0]
    return pc


def print_pc(shm):
    pc = read_pc(shm)
    print(f"PC = 0x{pc:x}")
    return pc


def setup_env(game):
    env = retro.make(game="SuperMarioWorld-Snes")
    _ = env.reset()
    return env


def step(env, shm):
    random_poke(env)
    obs, rew, done, info = env.step(env.action_space.sample())
    pc = print_pc(shm)
    env.render()
    if done:
        env.reset()
    return pc


def main():
    global VISITED
    signal.signal(signal.SIGINT, signal_handler)
    os.environ["RETRO_RUN_ID"] = "1337"
    shm = attach_shm()
    env = setup_env("SuperMarioWorld-Snes")
    while CONTINUE:
        pc = step(env, shm)
        VISITED.add(pc)
    return


if __name__ == "__main__":
    main()

