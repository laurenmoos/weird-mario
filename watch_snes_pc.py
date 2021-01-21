#! /usr/bin/env python

import sysv_ipc as ipc
import time
import os
import struct
import sys

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
    return

def main():
    shm = attach_shm()
    while True:
        print_pc(shm)
        time.sleep(0.1)
    return

if __name__ == "__main__":
    main()
