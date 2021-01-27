#! /usr/bin/env python

import signal
import json
import subprocess as sp
import time
import os
import struct
import sys
import retro
import sysv_ipc as ipc
import random
import pyDispel


DELAY = 0
CONTINUE = True
VISITED = set()
ROM = "SuperMarioWorld-Snes"

def get_disas_table(rom):
    rom_path = f"./retro/data/stable/{rom}/rom.sfc"
    return pyDispel.ingest(rom_path)


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


def setup_env(game):
    env = retro.make(game="SuperMarioWorld-Snes")
    _ = env.reset()
    return env


def step(env, shm):
    if len(VISITED) < 3:
        random_poke(env)
    obs, rew, done, info = env.step(env.action_space.sample())
    pc = read_pc(shm)
    info["pc"] = pc
    env.render()
    if done:
        env.reset()
    return pc


def main():
    global VISITED
    disas = get_disas_table(ROM)
    signal.signal(signal.SIGINT, signal_handler)
    os.environ["RETRO_RUN_ID"] = "1337"
    shm = attach_shm()
    env = setup_env(ROM)
    i = 3000
    while CONTINUE:
        pc = step(env, shm)
        try:
            dis, _instbytes = disas[pc]
            print(f"0x{pc:04x}:\t{dis}")
        except KeyError as e:
            print(f"No instruction found at 0x{pc:x}")

        VISITED.add(pc)
        i -= 1
        if i == 0:
            print("=== RESETTING ===")
            env.reset()
            VISITED = set()
            i = 3000
    return


if __name__ == "__main__":
    main()

