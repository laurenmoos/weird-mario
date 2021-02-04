#! /usr/bin/env python

import time
import sys
import retro
import random
import os

POKES='POKES' in os.environ
TRACE='NOTRACE' not in os.environ
DISALL='DISALL' in os.environ
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
    if POKES:
        random_poke(env)
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
    return info

def main():
    env = setup_env(ROM)
    i = CYCLE_LENGTH
    while CONTINUE:
        info = step(env)
        if TRACE:
            print(f"[+] {len(info['trace'])} instructions executed; {len(set(t for t in info['trace']))} distinct")
            if DISALL:
                for t in info['trace']:
                    print(t)
            else:
                for t in info['trace'][:10]:
                    #print(retro.dispel.disas_code(code=t[2], addr=t[0])[0])
                    print(env.disassemble(address=t[0], offset=t[1], bytecode=t[2]))
                print("...")
                for t in info['trace'][-10:]:
                    print(env.disassemble(address=t[0], offset=t[1], bytecode=t[2]))
        env.render()
        i -= 1
        if i == 0:
            print("=== RESETTING ===")
            env.reset()
            i = CYCLE_LENGTH
    return


if __name__ == "__main__":
    main()

