#! /usr/bin/env python

import time
import sys
import retro
import random


CYCLE_LENGTH = 500
CONTINUE = True
ROM = "SuperMarioWorld-Snes"


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


def setup_env(game):
    env = retro.make(game="SuperMarioWorld-Snes")
    _ = env.reset()
    return env


def step(env):
    random_poke(env)
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        env.reset()
    return info


def pp_trace(trace):
    if trace.bytes is None:
        print(f"0x{trace.addr:04x}: NO INSTRUCTION")
    else:
        print(f"0x{trace.addr:04x}\t{trace.inst}")

def main():
    env = setup_env(ROM)
    i = CYCLE_LENGTH
    while CONTINUE:
        info = step(env)
        print(f"[+] {len(info['trace'])} instructions executed; {len(set(t.addr for t in info['trace']))} distinct")
        for t in info['trace'][:10]:
            pp_trace(t)
        print("...")
        for pc in info['trace'][-10:]:
            pp_trace(t)
        i -= 1
        if i == 0:
            print("=== RESETTING ===")
            env.reset()
            i = CYCLE_LENGTH
    return


if __name__ == "__main__":
    main()

