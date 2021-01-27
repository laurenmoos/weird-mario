#! /usr/bin/env python3

import subprocess as sp
import json
import sys
import os


DISPEL_PATH = f"{os.path.dirname(__file__)}/dispel.exe"


def disas(rom):
    """Runs the dispel.exe binary on the rom path provided, and returns
    the output as a list of strings."""
    p = sp.Popen([DISPEL_PATH, rom], stdout=sp.PIPE)
    return [r.decode('utf-8') for r in p.stdout.readlines()]


def parse_addr(addr):
    """Should parse address of the format 80/DDA1 as 0x80DDA1."""
    return int(addr.replace("/", "").replace(":", ""), base=16)


def parse_bytes(hexstring):
    """Parse hexidecimal bytes."""
    return list(bytes.fromhex(hexstring.strip()))


def parse_line(line):
    """Parse a line of the disassembler output."""
    parts = line.strip().split('\t')
    if len(parts) == 2: # No instruction
        r_addr, r_ibytes = parts
        inst = "INVALID"
    else:
        r_addr, r_ibytes, inst = parts
    addr = parse_addr(r_addr)
    ibytes = parse_bytes(r_ibytes)
    return addr, ibytes, inst


def build_table(rows):
    """Build a lookup table mapping addresses to instructions."""
    table = dict()
    for row in rows:
        addr, ibytes, inst = parse_line(row)
        table[addr] = (inst, ibytes)
    return table


def ingest(rom):
    """Disassembles a ROM, provided by a path name, and returns a
    lookup table mapping addresses to instructions."""
    return build_table(disas(rom))


def export(rom):
    """Export the disassembly table of the ROM as a json file."""
    return json.dumps(ingest(rom))


if __name__ == "__main__":
    print(export(sys.argv[1]))
