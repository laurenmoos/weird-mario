#! /usr/bin/env python

import signal
import monkeyhex
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


CYCLE_LENGTH = 500
RUN_ID = random.randint(1,1<<31)
WORD_SIZE = 2
VISITED_BUFFER_SIZE = 1 << 15 * WORD_SIZE
DELAY = 0
CONTINUE = True
VISITED = set()
ROM = "SuperMarioWorld-Snes"
SHM = None

def get_disas_table(rom):
    rom_path = f"./retro/data/stable/{rom}/rom.sfc"
    return pyDispel.ingest(rom_path)


def signal_handler(sig, frame):
    global CONTINUE
    CONTINUE = False
    #print(f"SIGINT Received.\n{len(VISITED)} Addresses Visited:")
    #for addr in sorted(VISITED):
    #    print(f"    0x{addr:x}")
    ipc.remove_shared_memory(SHM.id)
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
    global SHM
    key = get_key()
    shm_size = VISITED_BUFFER_SIZE
    try:
        shm = ipc.SharedMemory(key, flags=ipc.IPC_CREX, mode=0o666, size=shm_size)
    except Exception as e:
        # TODO tighten up this except
        shm = ipc.SharedMemory(key, 0, 0)
        ipc.remove_shared_memory(shm.id)
        shm = ipc.SharedMemory(key, flags=ipc.IPC_CREX, mode=0o666, size=shm_size)

    shm.attach()
    SHM = shm
    return shm


def read_pc_vec(shm):
    count = struct.unpack("<H", shm.read(WORD_SIZE))[0]
    #print(f"# of addresses: 0x{count:x}")
    buf = shm.read((count+1) * WORD_SIZE)
    #print(f"count = {count}; expecting {(count+1)*WORD_SIZE}; len(buf) = {len(buf)}")
    addrs = list(struct.unpack(f"<{count+1}H", buf)[1:])
    return addrs


def read_pc(shm):
    buf = shm.read(4)
    pc = struct.unpack("<I", buf)[0]
    return pc


def setup_env(game):
    env = retro.make(game="SuperMarioWorld-Snes")
    _ = env.reset()
    return env


def step(env, shm):
    random_poke(env)
    obs, rew, done, info = env.step(env.action_space.sample())
    pc_vec = read_pc_vec(shm)
    info["pc_vec"] = pc_vec
    env.render()
    if done:
        env.reset()
    return pc_vec



def main():
    global VISITED
    disas = get_disas_table(ROM)
    signal.signal(signal.SIGINT, signal_handler)
    os.environ["RETRO_RUN_ID"] = f"{RUN_ID}"
    env = setup_env(ROM)
    shm = attach_shm()
    i = CYCLE_LENGTH
    while CONTINUE:
        pc_vec = step(env, shm)
        print(f"[+] {len(pc_vec)} instructions executed; {len(set(pc_vec))} unique")
        for pc in pc_vec:
            VISITED.add(pc)
        for pc in pc_vec[:10]:
            try:
                dis, _instbytes = disas[pc]
                print(f"0x{pc:04x}:\t{dis}")
            except KeyError as e:
                print(f"No instruction found at 0x{pc:x}")
        print("...")
        for pc in pc_vec[-10:]:
            try:
                dis, _instbytes = disas[pc]
                print(f"0x{pc:04x}:\t{dis}")
            except KeyError as e:
                print(f"No instruction found at 0x{pc:x}")
        i -= 1
        if i == 0:
            print("=== RESETTING ===")
            env.reset()
            VISITED = set()
            i = CYCLE_LENGTH
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Exception: {e}")
        try:
            ipc.remove_shared_memory(SHM)
        except Exception:
            pass

