import subprocess as sp
import json
import sys
import os

DISPEL_PATH = os.getenv("DISPEL_PATH")
if DISPEL_PATH is None:
    DISPEL_PATH = f"{os.path.dirname(__file__)}/../Dispel/dispel.exe"


def has_smc_header(rom):
    with open(rom, "rb") as f:
        hd = f.read(0x200)
    return not any(hd[4:])

def disas(rom):
    """Runs the dispel.exe binary on the rom path provided, and returns
    the output as a list of strings."""
    args = ['-x', '-a'] # 8-bit mode
    if has_smc_header(rom):
        print("SMC header detected")
        args.append('-n')
    else:
        print("No SMC header detected")
    out = []
    for flag in ['-i', '-s']:
        cmd = [DISPEL_PATH, *args, flag, rom]
        #skip_smc_header = '-n' if has_smc_header(rom) else ''
        p = sp.Popen(cmd, stderr=sp.PIPE, stdout=sp.PIPE)
        out += [r.decode('utf-8') for r in p.stdout.readlines()]
    return out

def parse_addr(addr):
    """Drop the bank number. We're assuming bank 0. Should parse address of the format 80/DDA1 as 0xDDA1."""
    # FIXME: Not entirely sure we're entitled to assume that the bank is always 0.
    # We might need to refine this assumption later.
    bank, addr = (int(x, base=16) for x in addr.replace(':','').split('/'))
    return bank, addr


def parse_bytes(hexstring):
    """Parse hexidecimal bytes."""
    return bytes.fromhex(hexstring.strip())


def parse_line(line):
    """Parse a line of the disassembler output."""
    parts = line.strip().split('\t')
    if len(parts) == 2: # No instruction
        r_addr, r_ibytes = parts
        inst = "INVALID"
    else:
        r_addr, r_ibytes, inst = parts
    bank, addr = parse_addr(r_addr)
    ibytes = parse_bytes(r_ibytes)
    return bank, addr, ibytes, inst


def build_table(rows):
    """Build a lookup table mapping addresses to instructions."""
    table = dict()
    for row in rows:
        bank, addr, ibytes, inst = parse_line(row)
        table[(bank, addr)] = (inst, ibytes)
    return table


def ingest(rom):
    """Disassembles a ROM, provided by a path name, and returns a
    lookup table mapping addresses to instructions."""
    return build_table(disas(rom))


def export(rom):
    """Export the disassembly table of the ROM as a json file."""
    return json.dumps(ingest(rom))

