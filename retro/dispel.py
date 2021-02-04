import subprocess as sp
from collections import namedtuple
import json
import tempfile
import sys
import os

DISPEL_PATH = os.getenv("DISPEL_PATH")
if DISPEL_PATH is None:
    DISPEL_PATH = f"{os.path.dirname(__file__)}/../Dispel/dispel.exe"

Trace = namedtuple('Trace', ['bank', 'addr', 'inst', 'bytes'])
#Trace.__repr__ = lambda t : \
#    f"Trace(addr={t.addr:04x}, inst='{t.inst}', bytes={bytes.hex(t.bytes, ' ')})"
Trace.__repr__ = lambda t : \
    f"{t.bank:02x}/{t.addr:04x}:    {bytes.hex(t.bytes, ' ') if t.bytes is not None else '':<15}{t.inst}"



OFFSETS = {
    0x0C: 3,
    0x0D: 3,
    0x0E: 3,
    0x1C: 3,
    0x20: 3,
    0x2C: 3,
    0x2D: 3,
    0x2E: 3,
    0x4C: 3,
    0x4D: 3,
    0x4E: 3,
    0x6D: 3,
    0x6E: 3,
    0x8C: 3,
    0x8D: 3,
    0x8E: 3,
    0x9C: 3,
    0xAC: 3,
    0xAD: 3,
    0xAE: 3,
    0xCC: 3,
    0xCD: 3,
    0xCE: 3,
    0xEC: 3,
    0xED: 3,
    0xEE: 3,
    0x7C: 3,
    0xFC: 3,
    0x1D: 3,
    0x1E: 3,
    0x3C: 3,
    0x3D: 3,
    0x3E: 3,
    0x5D: 3,
    0x5E: 3,
    0x7D: 3,
    0x7E: 3,
    0x9D: 3,
    0x9E: 3,
    0xBC: 3,
    0xBD: 3,
    0xDD: 3,
    0xDE: 3,
    0xFD: 3,
    0xFE: 3,
    0x19: 3,
    0x39: 3,
    0x59: 3,
    0x79: 3,
    0x99: 3,
    0xB9: 3,
    0xBE: 3,
    0xD9: 3,
    0xF9: 3,
    0x6C: 3,
    0xDC: 3,
    0x0F: 4,
    0x22: 4,
    0x2F: 4,
    0x4F: 4,
    0x5C: 4,
    0x6F: 4,
    0x8F: 4,
    0xAF: 4,
    0xCF: 4,
    0xEF: 4,
    0x1F: 4,
    0x3F: 4,
    0x5F: 4,
    0x7F: 4,
    0x9F: 4,
    0xBF: 4,
    0xDF: 4,
    0xFF: 4,
    0x0A: 1,
    0x1A: 1,
    0x2A: 1,
    0x3A: 1,
    0x4A: 1,
    0x6A: 1,
    0x44: 3,
    0x54: 3,
    0x04: 2,
    0x05: 2,
    0x06: 2,
    0x14: 2,
    0x24: 2,
    0x25: 2,
    0x26: 2,
    0x45: 2,
    0x46: 2,
    0x64: 2,
    0x65: 2,
    0x66: 2,
    0x84: 2,
    0x85: 2,
    0x86: 2,
    0xA4: 2,
    0xA5: 2,
    0xA6: 2,
    0xC4: 2,
    0xC5: 2,
    0xC6: 2,
    0xE4: 2,
    0xE5: 2,
    0xE6: 2,
    0x15: 2,
    0x16: 2,
    0x34: 2,
    0x35: 2,
    0x36: 2,
    0x55: 2,
    0x56: 2,
    0x74: 2,
    0x75: 2,
    0x76: 2,
    0x94: 2,
    0x95: 2,
    0xB4: 2,
    0xB5: 2,
    0xD5: 2,
    0xD6: 2,
    0xF5: 2,
    0xF6: 2,
    0x96: 2,
    0xB6: 2,
    0x12: 2,
    0x32: 2,
    0x52: 2,
    0x72: 2,
    0x92: 2,
    0xB2: 2,
    0xD2: 2,
    0xF2: 2,
    0x07: 2,
    0x27: 2,
    0x47: 2,
    0x67: 2,
    0x87: 2,
    0xA7: 2,
    0xC7: 2,
    0xE7: 2,
    0x01: 2,
    0x21: 2,
    0x41: 2,
    0x61: 2,
    0x81: 2,
    0xA1: 2,
    0xC1: 2,
    0xE1: 2,
    0x11: 2,
    0x31: 2,
    0x51: 2,
    0x71: 2,
    0x91: 2,
    0xB1: 2,
    0xD1: 2,
    0xF1: 2,
    0x17: 2,
    0x37: 2,
    0x57: 2,
    0x77: 2,
    0x97: 2,
    0xB7: 2,
    0xD7: 2,
    0xF7: 2,
    0x28: 1,
    0x2B: 1,
    0x68: 1,
    0x7A: 1,
    0xAB: 1,
    0xFA: 1,
    0x08: 1,
    0x0B: 1,
    0x48: 1,
    0x4B: 1,
    0x5A: 1,
    0x8B: 1,
    0xDA: 1,
    0x6B: 1,
    0x60: 1,
    0x40: 1,
    0x18: 1,
    0x1B: 1,
    0x38: 1,
    0x3B: 1,
    0x58: 1,
    0x5B: 1,
    0x78: 1,
    0x7B: 1,
    0x88: 1,
    0x8A: 1,
    0x98: 1,
    0x9A: 1,
    0x9B: 1,
    0xA8: 1,
    0xAA: 1,
    0xB8: 1,
    0xBA: 1,
    0xBB: 1,
    0xC8: 1,
    0xCA: 1,
    0xCB: 1,
    0xD8: 1,
    0xDB: 1,
    0xE8: 1,
    0xEA: 1,
    0xEB: 1,
    0xF8: 1,
    0xFB: 1,
    0x10: 2,
    0x30: 2,
    0x50: 2,
    0x70: 2,
    0x80: 2,
    0x90: 2,
    0xB0: 2,
    0xD0: 2,
    0xF0: 2,
    0x62: 3,
    0x82: 3,
    0x13: 2,
    0x33: 2,
    0x53: 2,
    0x73: 2,
    0x93: 2,
    0xB3: 2,
    0xD3: 2,
    0xF3: 2,
    0xF4: 3,
    0xD4: 2,
    0x03: 2,
    0x23: 2,
    0x43: 2,
    0x63: 2,
    0x83: 2,
    0xA3: 2,
    0xC3: 2,
    0xE3: 2,
    0x42: 2,
    0x00: 2,
    0x02: 2,
    0xC2: 2,
    0xE2: 2,
    ### these should have offset 2 if flag & 0x20, 3 otherwise
    0x09: lambda flag: 2 if flag & 0x20 else 3,
    0x29: lambda flag: 2 if flag & 0x20 else 3,
    0x49: lambda flag: 2 if flag & 0x20 else 3,
    0x69: lambda flag: 2 if flag & 0x20 else 3,
    0x89: lambda flag: 2 if flag & 0x20 else 3,
    0xA9: lambda flag: 2 if flag & 0x20 else 3,
    0xC9: lambda flag: 2 if flag & 0x20 else 3,
    0xE9: lambda flag: 2 if flag & 0x20 else 3,
    # these should have offset 2 if flag & 0x10, 3 otherwise
    0xA0: lambda flag: 2 if flag & 0x10 else 3,
    0xA2: lambda flag: 2 if flag & 0x10 else 3,
    0xC0: lambda flag: 2 if flag & 0x10 else 3,
    0xE0: lambda flag: 2 if flag & 0x10 else 3,
}

def get_offset(inst, flag):
    try:
        n = OFFSETS[inst[0]]
    except KeyError:
        return 1
    if type(n) is int:
        return n
    else:
        return n(flag)


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
    return Trace(bank,  addr, inst, bytes=ibytes) 


def build_table(rows):
    """Build a lookup table mapping addresses to instructions."""
    table = dict()
    for row in rows:
        trace = parse_line(row)
        table[(trace.bank, trace.addr)] = trace
    return table


def ingest(rom):
    """Disassembles a ROM, provided by a path name, and returns a
    lookup table mapping addresses to instructions."""
    return build_table(disas(rom))


def export(rom):
    """Export the disassembly table of the ROM as a json file."""
    return json.dumps(ingest(rom))


def disas_code(code, addr=0, flag=0):
    """Slow, use only for debugging."""
    args = ['-i', '-L', f'{len(code):X}', '-g', f'{addr:06X}']
    if flag & 0x20:
        args.append('-x')
    if flag & 0x10:
        args.append('-a')
    cmd = [DISPEL_PATH, *args, '/dev/stdin']
    p = sp.Popen(cmd, stdout=sp.PIPE, stdin=sp.PIPE)
    p.stdin.write(code)
    p.stdin.close()
    out = p.stdout.readlines()
    return [parse_line(r.decode('utf-8')) for r in out]


