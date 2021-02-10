/***********************************************************************************
  Snes9x - Portable Super Nintendo Entertainment System (TM) emulator.

  (c) Copyright 1996 - 2002  Gary Henderson (gary.henderson@ntlworld.com),
                             Jerremy Koot (jkoot@snes9x.com)

  (c) Copyright 2002 - 2004  Matthew Kendora

  (c) Copyright 2002 - 2005  Peter Bortas (peter@bortas.org)

  (c) Copyright 2004 - 2005  Joel Yliluoma (http://iki.fi/bisqwit/)

  (c) Copyright 2001 - 2006  John Weidman (jweidman@slip.net)

  (c) Copyright 2002 - 2006  funkyass (funkyass@spam.shaw.ca),
                             Kris Bleakley (codeviolation@hotmail.com)

  (c) Copyright 2002 - 2010  Brad Jorsch (anomie@users.sourceforge.net),
                             Nach (n-a-c-h@users.sourceforge.net),

  (c) Copyright 2002 - 2011  zones (kasumitokoduck@yahoo.com)

  (c) Copyright 2006 - 2007  nitsuja

  (c) Copyright 2009 - 2011  BearOso,
                             OV2

  (c) Copyright 2011 - 2016  Hans-Kristian Arntzen,
                             Daniel De Matteis
                             (Under no circumstances will commercial rights be given)


  BS-X C emulator code
  (c) Copyright 2005 - 2006  Dreamer Nom,
                             zones

  C4 x86 assembler and some C emulation code
  (c) Copyright 2000 - 2003  _Demo_ (_demo_@zsnes.com),
                             Nach,
                             zsKnight (zsknight@zsnes.com)

  C4 C++ code
  (c) Copyright 2003 - 2006  Brad Jorsch,
                             Nach

  DSP-1 emulator code
  (c) Copyright 1998 - 2006  _Demo_,
                             Andreas Naive (andreasnaive@gmail.com),
                             Gary Henderson,
                             Ivar (ivar@snes9x.com),
                             John Weidman,
                             Kris Bleakley,
                             Matthew Kendora,
                             Nach,
                             neviksti (neviksti@hotmail.com)

  DSP-2 emulator code
  (c) Copyright 2003         John Weidman,
                             Kris Bleakley,
                             Lord Nightmare (lord_nightmare@users.sourceforge.net),
                             Matthew Kendora,
                             neviksti

  DSP-3 emulator code
  (c) Copyright 2003 - 2006  John Weidman,
                             Kris Bleakley,
                             Lancer,
                             z80 gaiden

  DSP-4 emulator code
  (c) Copyright 2004 - 2006  Dreamer Nom,
                             John Weidman,
                             Kris Bleakley,
                             Nach,
                             z80 gaiden

  OBC1 emulator code
  (c) Copyright 2001 - 2004  zsKnight,
                             pagefault (pagefault@zsnes.com),
                             Kris Bleakley
                             Ported from x86 assembler to C by sanmaiwashi

  SPC7110 and RTC C++ emulator code used in 1.39-1.51
  (c) Copyright 2002         Matthew Kendora with research by
                             zsKnight,
                             John Weidman,
                             Dark Force

  SPC7110 and RTC C++ emulator code used in 1.52+
  (c) Copyright 2009         byuu,
                             neviksti

  S-DD1 C emulator code
  (c) Copyright 2003         Brad Jorsch with research by
                             Andreas Naive,
                             John Weidman

  S-RTC C emulator code
  (c) Copyright 2001 - 2006  byuu,
                             John Weidman

  ST010 C++ emulator code
  (c) Copyright 2003         Feather,
                             John Weidman,
                             Kris Bleakley,
                             Matthew Kendora

  Super FX x86 assembler emulator code
  (c) Copyright 1998 - 2003  _Demo_,
                             pagefault,
                             zsKnight

  Super FX C emulator code
  (c) Copyright 1997 - 1999  Ivar,
                             Gary Henderson,
                             John Weidman

  Sound emulator code used in 1.5-1.51
  (c) Copyright 1998 - 2003  Brad Martin
  (c) Copyright 1998 - 2006  Charles Bilyue'

  Sound emulator code used in 1.52+
  (c) Copyright 2004 - 2007  Shay Green (gblargg@gmail.com)

  SH assembler code partly based on x86 assembler code
  (c) Copyright 2002 - 2004  Marcus Comstedt (marcus@mc.pp.se)

  2xSaI filter
  (c) Copyright 1999 - 2001  Derek Liauw Kie Fa

  HQ2x, HQ3x, HQ4x filters
  (c) Copyright 2003         Maxim Stepin (maxim@hiend3d.com)

  NTSC filter
  (c) Copyright 2006 - 2007  Shay Green

  GTK+ GUI code
  (c) Copyright 2004 - 2011  BearOso

  Win32 GUI code
  (c) Copyright 2003 - 2006  blip,
                             funkyass,
                             Matthew Kendora,
                             Nach,
                             nitsuja
  (c) Copyright 2009 - 2011  OV2

  Mac OS GUI code
  (c) Copyright 1998 - 2001  John Stiles
  (c) Copyright 2001 - 2011  zones

  Libretro port
  (c) Copyright 2011 - 2016  Hans-Kristian Arntzen,
                             Daniel De Matteis
                             (Under no circumstances will commercial rights be given)


  Specific ports contains the works of other authors. See headers in
  individual files.


  Snes9x homepage: http://www.snes9x.com/

  Permission to use, copy, modify and/or distribute Snes9x in both binary
  and source form, for non-commercial purposes, is hereby granted without
  fee, providing that this license information and copyright notice appear
  with all copies and any derived work.

  This software is provided 'as-is', without any express or implied
  warranty. In no event shall the authors be held liable for any damages
  arising from the use of this software or it's derivatives.

  Snes9x is freeware for PERSONAL USE only. Commercial users should
  seek permission of the copyright holders first. Commercial use includes,
  but is not limited to, charging money for Snes9x or software derived from
  Snes9x, including Snes9x or derivatives in commercial game bundles, and/or
  using Snes9x as a promotion for your commercial product.

  The copyright holders request that bug fixes and improvements to the code
  should be forwarded to them so everyone can benefit from the modifications
  in future versions.

  Super NES and Super Nintendo Entertainment System are trademarks of
  Nintendo Co., Limited and its subsidiary companies.
 ***********************************************************************************/

#define CPU_OPCODE_INSTRUMENTATION 1
#define VISITED_BUFFER_SIZE (1 << 15)

#include "snes9x.h"
#include "memmap.h"
#include "cpuops.h"
#include "dma.h"
#include "apu/apu.h"
#include "fxemu.h"
// #include "debug.h"
#include "snapshot.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#ifdef DEBUGGER
#include "debug.h"
#include "missing.h"
#endif

// NOTE: we could contain the entire instruction (in bytecode) in the upper half of a long integer
// That might be the most efficient way of sending it over memory.
// though we lose the proc flags then

typedef uint64 Word;

int inst_offset(unsigned char *mem, unsigned int flag)
{
	int offset = 0;
    switch (mem[0]) {
        // Absolute
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x1C:
    case 0x20:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x6D:
    case 0x6E:
    case 0x8C:
    case 0x8D:
    case 0x8E:
    case 0x9C:
    case 0xAC:
    case 0xAD:
    case 0xAE:
    case 0xCC:
    case 0xCD:
    case 0xCE:
    case 0xEC:
    case 0xED:
    case 0xEE:
        offset=3;
        break;
        // Absolute Indexed Indirect
    case 0x7C:
    case 0xFC:
        offset=3;
        break;
        // Absolute Indexed, X
    case 0x1D:
    case 0x1E:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x5D:
    case 0x5E:
    case 0x7D:
    case 0x7E:
    case 0x9D:
    case 0x9E:
    case 0xBC:
    case 0xBD:
    case 0xDD:
    case 0xDE:
    case 0xFD:
    case 0xFE:
        offset=3;
        break;
        // Absolute Indexed, Y
    case 0x19:
    case 0x39:
    case 0x59:
    case 0x79:
    case 0x99:
    case 0xB9:
    case 0xBE:
    case 0xD9:
    case 0xF9:
        offset=3;
        break;
        // Absolute Indirect
    case 0x6C:
        offset=3;
        break;
        // Absolute Indirect Long
    case 0xDC:
        offset=3;
        break;
        // Absolute Long
    case 0x0F:
    case 0x22:
    case 0x2F:
    case 0x4F:
    case 0x5C:
    case 0x6F:
    case 0x8F:
    case 0xAF:
    case 0xCF:
    case 0xEF:
        offset=4;
        break;
        // Absolute Long Indexed, X
    case 0x1F:
    case 0x3F:
    case 0x5F:
    case 0x7F:
    case 0x9F:
    case 0xBF:
    case 0xDF:
    case 0xFF:
        offset=4;
        break;
        // Accumulator
    case 0x0A:
    case 0x1A:
    case 0x2A:
    case 0x3A:
    case 0x4A:
    case 0x6A:
        offset=1;
        break;
        // Block Move
    case 0x44:
    case 0x54:
        offset=3;
        break;
        // Direct Page
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x14:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x45:
    case 0x46:
    case 0x64:
    case 0x65:
    case 0x66:
    case 0x84:
    case 0x85:
    case 0x86:
    case 0xA4:
    case 0xA5:
    case 0xA6:
    case 0xC4:
    case 0xC5:
    case 0xC6:
    case 0xE4:
    case 0xE5:
    case 0xE6:
        offset=2;
        break;
        // Direct Page Indexed, X
    case 0x15:
    case 0x16:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x55:
    case 0x56:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x94:
    case 0x95:
    case 0xB4:
    case 0xB5:
    case 0xD5:
    case 0xD6:
    case 0xF5:
    case 0xF6:
        offset=2;
        break;
        // Direct Page Indexed, Y
    case 0x96:
    case 0xB6:
        offset=2;
        break;
        // Direct Page Indirect
    case 0x12:
    case 0x32:
    case 0x52:
    case 0x72:
    case 0x92:
    case 0xB2:
    case 0xD2:
    case 0xF2:
        offset=2;
        break;
        // Direct Page Indirect Long
    case 0x07:
    case 0x27:
    case 0x47:
    case 0x67:
    case 0x87:
    case 0xA7:
    case 0xC7:
    case 0xE7:
        offset=2;
        break;
        // Direct Page Indexed Indirect, X
    case 0x01:
    case 0x21:
    case 0x41:
    case 0x61:
    case 0x81:
    case 0xA1:
    case 0xC1:
    case 0xE1:
        offset=2;
        break;
        // Direct Page Indirect Indexed, Y
    case 0x11:
    case 0x31:
    case 0x51:
    case 0x71:
    case 0x91:
    case 0xB1:
    case 0xD1:
    case 0xF1:
        offset=2;
        break;
        // Direct Page Indirect Long Indexed, Y
    case 0x17:
    case 0x37:
    case 0x57:
    case 0x77:
    case 0x97:
    case 0xB7:
    case 0xD7:
    case 0xF7:
        offset=2;
        break;
        // Stack (Pull)
    case 0x28:
    case 0x2B:
    case 0x68:
    case 0x7A:
    case 0xAB:
    case 0xFA:
        // Stack (Push)
    case 0x08:
    case 0x0B:
    case 0x48:
    case 0x4B:
    case 0x5A:
    case 0x8B:
    case 0xDA:
        // Stack (RTL)
    case 0x6B:
        // Stack (RTS)
    case 0x60:
        // Stack/RTI
    case 0x40:
        // Implied
    case 0x18:
    case 0x1B:
    case 0x38:
    case 0x3B:
    case 0x58:
    case 0x5B:
    case 0x78:
    case 0x7B:
    case 0x88:
    case 0x8A:
    case 0x98:
    case 0x9A:
    case 0x9B:
    case 0xA8:
    case 0xAA:
    case 0xB8:
    case 0xBA:
    case 0xBB:
    case 0xC8:
    case 0xCA:
    case 0xCB:
    case 0xD8:
    case 0xDB:
    case 0xE8:
    case 0xEA:
    case 0xEB:
    case 0xF8:
    case 0xFB:
        offset = 1;
        break;
        // Program Counter Relative
    case 0x10:
    case 0x30:
    case 0x50:
    case 0x70:
    case 0x80:
    case 0x90:
    case 0xB0:
    case 0xD0:
    case 0xF0:
        // Calculate the signed value of the param
        offset = 2;
        break;
        // Stack (Program Counter Relative Long)
    case 0x62:
        // Program Counter Relative Long
    case 0x82:
        // Calculate the signed value of the param
        offset = 3;
        break;
        // Stack Relative Indirect Indexed, Y
    case 0x13:
    case 0x33:
    case 0x53:
    case 0x73:
    case 0x93:
    case 0xB3:
    case 0xD3:
    case 0xF3:
        offset = 2;
        break;
        // Stack (Absolute)
    case 0xF4:
        offset = 3;
        break;
        // Stack (Direct Page Indirect)
    case 0xD4:
        offset = 2;
        break;
        offset = 3;
        break;
        // Stack Relative
    case 0x03:
    case 0x23:
    case 0x43:
    case 0x63:
    case 0x83:
    case 0xA3:
    case 0xC3:
    case 0xE3:
        offset = 2;
        break;
        // WDM mode
    case 0x42:
        // Stack/Interrupt
    case 0x00:
    case 0x02:
        offset = 2;
        break;
        // Immediate (Invariant)
    case 0xC2:
        // REP following
        offset = 2;
        break;
    case 0xE2:
        // SEP following
        offset = 2;
        break;
        // Immediate (A size dependent)
    case 0x09:
    case 0x29:
    case 0x49:
    case 0x69:
    case 0x89:
    case 0xA9:
    case 0xC9:
    case 0xE9:
        if (flag & 0x20)
        {
            offset = 2;
        }
        else
        {
            offset = 3;
        }
        break;
        // Immediate (X/Y size dependent)
    case 0xA0:
    case 0xA2:
    case 0xC0:
    case 0xE0:
        if (flag & 0x10)
        {
            offset = 2;
        }
        else
        {
            offset = 3;
        }
        break;
    default:
        fprintf(stderr, "[!] WARNING: Unhandled Addressing Mode: %02X\n",mem[0]);
    }
	return offset;
}

// Disassembler
int disasm(const unsigned char *mem, unsigned long pos, unsigned short int flag, char *inst, unsigned char tsrc)
{
    // temp buffers to hold instruction,parameters and hex
    char ibuf[5],pbuf[20],hbuf[9];
    // variables to hold the instruction increment and signed params
    int offset,sval,i;

    // Parse out instruction mnemonic

    switch (mem[0])
    {
        // ADC
    case 0x69:
    case 0x6D:
    case 0x6F:
    case 0x65:
    case 0x72:
    case 0x67:
    case 0x7D:
    case 0x7F:
    case 0x79:
    case 0x75:
    case 0x61:
    case 0x71:
    case 0x77:
    case 0x63:
    case 0x73:
        strcpy(ibuf,"adc");
        break;
        // AND
    case 0x29:
    case 0x2D:
    case 0x2F:
    case 0x25:
    case 0x32:
    case 0x27:
    case 0x3D:
    case 0x3F:
    case 0x39:
    case 0x35:
    case 0x21:
    case 0x31:
    case 0x37:
    case 0x23:
    case 0x33:
        strcpy(ibuf,"and");
        break;
        // ASL
    case 0x0A:
    case 0x0E:
    case 0x06:
    case 0x1E:
    case 0x16:
        strcpy(ibuf,"asl");
        break;
        // BCC
    case 0x90:
        strcpy(ibuf,"bcc");
        break;
        // BCS
    case 0xB0:
        strcpy(ibuf,"bcs");
        break;
        // BEQ
    case 0xF0:
        strcpy(ibuf,"beq");
        break;
        // BNE
    case 0xD0:
        strcpy(ibuf,"bne");
        break;
        // BMI
    case 0x30:
        strcpy(ibuf,"bmi");
        break;
        // BPL
    case 0x10:
        strcpy(ibuf,"bpl");
        break;
        // BVC
    case 0x50:
        strcpy(ibuf,"bvc");
        break;
        // BVS
    case 0x70:
        strcpy(ibuf,"bvs");
        break;
        // BRA
    case 0x80:
        strcpy(ibuf,"bra");
        break;
        // BRL
    case 0x82:
        strcpy(ibuf,"brl");
        break;
        // BIT
    case 0x89:
    case 0x2C:
    case 0x24:
    case 0x3C:
    case 0x34:
        strcpy(ibuf,"bit");
        break;
        // BRK
    case 0x00:
        strcpy(ibuf,"brk");
        break;
        // CLC
    case 0x18:
        strcpy(ibuf,"clc");
        break;
        // CLD
    case 0xD8:
        strcpy(ibuf,"cld");
        break;
        // CLI
    case 0x58:
        strcpy(ibuf,"cli");
        break;
        // CLV
    case 0xB8:
        strcpy(ibuf,"clv");
        break;
        // SEC
    case 0x38:
        strcpy(ibuf,"sec");
        break;
        // SED
    case 0xF8:
        strcpy(ibuf,"sed");
        break;
        // SEI
    case 0x78:
        strcpy(ibuf,"sei");
        break;
        // CMP
    case 0xC9:
    case 0xCD:
    case 0xCF:
    case 0xC5:
    case 0xD2:
    case 0xC7:
    case 0xDD:
    case 0xDF:
    case 0xD9:
    case 0xD5:
    case 0xC1:
    case 0xD1:
    case 0xD7:
    case 0xC3:
    case 0xD3:
        strcpy(ibuf,"cmp");
        break;
        // COP
    case 0x02:
        strcpy(ibuf,"cop");
        break;
        // CPX
    case 0xE0:
    case 0xEC:
    case 0xE4:
        strcpy(ibuf,"cpx");
        break;
        // CPY
    case 0xC0:
    case 0xCC:
    case 0xC4:
        strcpy(ibuf,"cpy");
        break;
        // DEC
    case 0x3A:
    case 0xCE:
    case 0xC6:
    case 0xDE:
    case 0xD6:
        strcpy(ibuf,"dec");
        break;
        // DEX
    case 0xCA:
        strcpy(ibuf,"dex");
        break;
        // DEY
    case 0x88:
        strcpy(ibuf,"dey");
        break;
        // EOR
    case 0x49:
    case 0x4D:
    case 0x4F:
    case 0x45:
    case 0x52:
    case 0x47:
    case 0x5D:
    case 0x5F:
    case 0x59:
    case 0x55:
    case 0x41:
    case 0x51:
    case 0x57:
    case 0x43:
    case 0x53:
        strcpy(ibuf,"eor");
        break;
        // INC
    case 0x1A:
    case 0xEE:
    case 0xE6:
    case 0xFE:
    case 0xF6:
        strcpy(ibuf,"inc");
        break;
        // INX
    case 0xE8:
        strcpy(ibuf,"inx");
        break;
        // INY
    case 0xC8:
        strcpy(ibuf,"iny");
        break;
        // JMP
    case 0x4C:
    case 0x6C:
    case 0x7C:
    case 0x5C:
    case 0xDC:
        strcpy(ibuf,"jmp");
        break;
        // JSR
    case 0x22:
    case 0x20:
    case 0xFC:
        strcpy(ibuf,"jsr");
        break;
        // LDA
    case 0xA9:
    case 0xAD:
    case 0xAF:
    case 0xA5:
    case 0xB2:
    case 0xA7:
    case 0xBD:
    case 0xBF:
    case 0xB9:
    case 0xB5:
    case 0xA1:
    case 0xB1:
    case 0xB7:
    case 0xA3:
    case 0xB3:
        strcpy(ibuf,"lda");
        break;
        // LDX
    case 0xA2:
    case 0xAE:
    case 0xA6:
    case 0xBE:
    case 0xB6:
        strcpy(ibuf,"ldx");
        break;
        // LDY
    case 0xA0:
    case 0xAC:
    case 0xA4:
    case 0xBC:
    case 0xB4:
        strcpy(ibuf,"ldy");
        break;
        // LSR
    case 0x4A:
    case 0x4E:
    case 0x46:
    case 0x5E:
    case 0x56:
        strcpy(ibuf,"lsr");
        break;
        // MVN
    case 0x54:
        strcpy(ibuf,"mvn");
        break;
        // MVP
    case 0x44:
        strcpy(ibuf,"mvp");
        break;
        // NOP
    case 0xEA:
        strcpy(ibuf,"nop");
        break;
        // ORA
    case 0x09:
    case 0x0D:
    case 0x0F:
    case 0x05:
    case 0x12:
    case 0x07:
    case 0x1D:
    case 0x1F:
    case 0x19:
    case 0x15:
    case 0x01:
    case 0x11:
    case 0x17:
    case 0x03:
    case 0x13:
        strcpy(ibuf,"ora");
        break;
        // PEA
    case 0xF4:
        strcpy(ibuf,"pea");
        break;
        // PEI
    case 0xD4:
        strcpy(ibuf,"pei");
        break;
        // PER
    case 0x62:
        strcpy(ibuf,"per");
        break;
        // PHA
    case 0x48:
        strcpy(ibuf,"pha");
        break;
        // PHP
    case 0x08:
        strcpy(ibuf,"php");
        break;
        // PHX
    case 0xDA:
        strcpy(ibuf,"phx");
        break;
        // PHY
    case 0x5A:
        strcpy(ibuf,"phy");
        break;
        // PLA
    case 0x68:
        strcpy(ibuf,"pla");
        break;
        // PLP
    case 0x28:
        strcpy(ibuf,"plp");
        break;
        // PLX
    case 0xFA:
        strcpy(ibuf,"plx");
        break;
        // PLY
    case 0x7A:
        strcpy(ibuf,"ply");
        break;
        // PHB
    case 0x8B:
        strcpy(ibuf,"phb");
        break;
        // PHD
    case 0x0B:
        strcpy(ibuf,"phd");
        break;
        // PHK
    case 0x4B:
        strcpy(ibuf,"phk");
        break;
        // PLB
    case 0xAB:
        strcpy(ibuf,"plb");
        break;
        // PLD
    case 0x2B:
        strcpy(ibuf,"pld");
        break;
        // REP
    case 0xC2:
        strcpy(ibuf,"rep");
        break;
        // ROL
    case 0x2A:
    case 0x2E:
    case 0x26:
    case 0x3E:
    case 0x36:
        strcpy(ibuf,"rol");
        break;
        // ROR
    case 0x6A:
    case 0x6E:
    case 0x66:
    case 0x7E:
    case 0x76:
        strcpy(ibuf,"ror");
        break;
        // RTI
    case 0x40:
        strcpy(ibuf,"rti");
        if(tsrc&0x2)
            strcat(ibuf,"\n");
        break;
        // RTL
    case 0x6B:
        strcpy(ibuf,"rtl");
        if(tsrc&0x2)
            strcat(ibuf,"\n");
        break;
        // RTS
    case 0x60:
        strcpy(ibuf,"rts");
        if(tsrc&0x2)
            strcat(ibuf,"\n");
        break;
        // SBC
    case 0xE9:
    case 0xED:
    case 0xEF:
    case 0xE5:
    case 0xF2:
    case 0xE7:
    case 0xFD:
    case 0xFF:
    case 0xF9:
    case 0xF5:
    case 0xE1:
    case 0xF1:
    case 0xF7:
    case 0xE3:
    case 0xF3:
        strcpy(ibuf,"sbc");
        break;
        // SEP
    case 0xE2:
        strcpy(ibuf,"sep");
        break;
        // STA
    case 0x8D:
    case 0x8F:
    case 0x85:
    case 0x92:
    case 0x87:
    case 0x9D:
    case 0x9F:
    case 0x99:
    case 0x95:
    case 0x81:
    case 0x91:
    case 0x97:
    case 0x83:
    case 0x93:
        strcpy(ibuf,"sta");
        break;
        // STP
    case 0xDB:
        strcpy(ibuf,"stp");
        break;
        // STX
    case 0x8E:
    case 0x86:
    case 0x96:
        strcpy(ibuf,"stx");
        break;
        // STY
    case 0x8C:
    case 0x84:
    case 0x94:
        strcpy(ibuf,"sty");
        break;
        // STZ
    case 0x9C:
    case 0x64:
    case 0x9E:
    case 0x74:
        strcpy(ibuf,"stz");
        break;
        // TAX
    case 0xAA:
        strcpy(ibuf,"tax");
        break;
        // TAY
    case 0xA8:
        strcpy(ibuf,"tay");
        break;
        // TXA
    case 0x8A:
        strcpy(ibuf,"txa");
        break;
        // TYA
    case 0x98:
        strcpy(ibuf,"tya");
        break;
        // TSX
    case 0xBA:
        strcpy(ibuf,"tsx");
        break;
        // TXS
    case 0x9A:
        strcpy(ibuf,"txs");
        break;
        // TXY
    case 0x9B:
        strcpy(ibuf,"txy");
        break;
        // TYX
    case 0xBB:
        strcpy(ibuf,"tyx");
        break;
        // TCD
    case 0x5B:
        strcpy(ibuf,"tcd");
        break;
        // TDC
    case 0x7B:
        strcpy(ibuf,"tdc");
        break;
        // TCS
    case 0x1B:
        strcpy(ibuf,"tcs");
        break;
        // TSC
    case 0x3B:
        strcpy(ibuf,"tsc");
        break;
        // TRB
    case 0x1C:
    case 0x14:
        strcpy(ibuf,"trb");
        break;
        // TSB
    case 0x0C:
    case 0x04:
        strcpy(ibuf,"tsb");
        break;
        // WAI
    case 0xCB:
        strcpy(ibuf,"wai");
        break;
        // WDM
    case 0x42:
        strcpy(ibuf,"wdm");
        break;
        // XBA
    case 0xEB:
        strcpy(ibuf,"xba");
        break;
        // XCE
    case 0xFB:
        strcpy(ibuf,"xce");
        break;
    default:
        // Illegal
        printf("Unhandled instruction: %02X\n",mem[0]);
        exit(1);
    };

    // Parse out parameter lis    // Parse out parameter list
    switch(mem[0]){
        // Absolute
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x1C:
    case 0x20:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x6D:
    case 0x6E:
    case 0x8C:
    case 0x8D:
    case 0x8E:
    case 0x9C:
    case 0xAC:
    case 0xAD:
    case 0xAE:
    case 0xCC:
    case 0xCD:
    case 0xCE:
    case 0xEC:
    case 0xED:
    case 0xEE:
        sprintf(pbuf,"$%04X",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indexed Indirect
    case 0x7C:
    case 0xFC:
        sprintf(pbuf,"($%04X,X)",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indexed, X
    case 0x1D:
    case 0x1E:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x5D:
    case 0x5E:
    case 0x7D:
    case 0x7E:
    case 0x9D:
    case 0x9E:
    case 0xBC:
    case 0xBD:
    case 0xDD:
    case 0xDE:
    case 0xFD:
    case 0xFE:
        sprintf(pbuf,"$%04X,X",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indexed, Y
    case 0x19:
    case 0x39:
    case 0x59:
    case 0x79:
    case 0x99:
    case 0xB9:
    case 0xBE:
    case 0xD9:
    case 0xF9:
        sprintf(pbuf,"$%04X,Y",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indirect
    case 0x6C:
        sprintf(pbuf,"($%04X)",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indirect Long
    case 0xDC:
        sprintf(pbuf,"[$%04X]",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Long
    case 0x0F:
    case 0x22:
    case 0x2F:
    case 0x4F:
    case 0x5C:
    case 0x6F:
    case 0x8F:
    case 0xAF:
    case 0xCF:
    case 0xEF:
        sprintf(pbuf,"$%06X",mem[1]+mem[2]*256+mem[3]*65536);
        offset=4;
        break;
        // Absolute Long Indexed, X
    case 0x1F:
    case 0x3F:
    case 0x5F:
    case 0x7F:
    case 0x9F:
    case 0xBF:
    case 0xDF:
    case 0xFF:
        sprintf(pbuf,"$%06X,X",mem[1]+mem[2]*256+mem[3]*65536);
        offset=4;
        break;
        // Accumulator
    case 0x0A:
    case 0x1A:
    case 0x2A:
    case 0x3A:
    case 0x4A:
    case 0x6A:
        sprintf(pbuf,"A");
        offset=1;
        break;
        // Block Move
    case 0x44:
    case 0x54:
        sprintf(pbuf,"$%02X,$%02X",mem[1],mem[2]);
        offset=3;
        break;
        // Direct Page
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x14:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x45:
    case 0x46:
    case 0x64:
    case 0x65:
    case 0x66:
    case 0x84:
    case 0x85:
    case 0x86:
    case 0xA4:
    case 0xA5:
    case 0xA6:
    case 0xC4:
    case 0xC5:
    case 0xC6:
    case 0xE4:
    case 0xE5:
    case 0xE6:
        sprintf(pbuf,"$%02X",mem[1]);
        offset=2;
        break;
        // Direct Page Indexed, X
    case 0x15:
    case 0x16:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x55:
    case 0x56:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x94:
    case 0x95:
    case 0xB4:
    case 0xB5:
    case 0xD5:
    case 0xD6:
    case 0xF5:
    case 0xF6:
        sprintf(pbuf,"$%02X,X",mem[1]);
        offset=2;
        break;
        // Direct Page Indexed, Y
    case 0x96:
    case 0xB6:
        sprintf(pbuf,"$%02X,Y",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect
    case 0x12:
    case 0x32:
    case 0x52:
    case 0x72:
    case 0x92:
    case 0xB2:
    case 0xD2:
    case 0xF2:
        sprintf(pbuf,"($%02X)",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect Long
    case 0x07:
    case 0x27:
    case 0x47:
    case 0x67:
    case 0x87:
    case 0xA7:
    case 0xC7:
    case 0xE7:
        sprintf(pbuf,"[$%02X]",mem[1]);
        offset=2;
        break;
        // Direct Page Indexed Indirect, X
    case 0x01:
    case 0x21:
    case 0x41:
    case 0x61:
    case 0x81:
    case 0xA1:
    case 0xC1:
    case 0xE1:
        sprintf(pbuf,"($%02X,X)",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect Indexed, Y
    case 0x11:
    case 0x31:
    case 0x51:
    case 0x71:
    case 0x91:
    case 0xB1:
    case 0xD1:
    case 0xF1:
        sprintf(pbuf,"($%02X),Y",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect Long Indexed, Y
    case 0x17:
    case 0x37:
    case 0x57:
    case 0x77:
    case 0x97:
    case 0xB7:
    case 0xD7:
    case 0xF7:
        sprintf(pbuf,"[$%02X],Y",mem[1]);
        offset=2;
        break;
        // Stack (Pull)
    case 0x28:
    case 0x2B:
    case 0x68:
    case 0x7A:
    case 0xAB:
    case 0xFA:
        // Stack (Push)
    case 0x08:
    case 0x0B:
    case 0x48:
    case 0x4B:
    case 0x5A:
    case 0x8B:
    case 0xDA:
        // Stack (RTL)
    case 0x6B:
        // Stack (RTS)
    case 0x60:
        // Stack/RTI
    case 0x40:
        // Implied
    case 0x18:
    case 0x1B:
    case 0x38:
    case 0x3B:
    case 0x58:
    case 0x5B:
    case 0x78:
    case 0x7B:
    case 0x88:
    case 0x8A:
    case 0x98:
    case 0x9A:
    case 0x9B:
    case 0xA8:
    case 0xAA:
    case 0xB8:
    case 0xBA:
    case 0xBB:
    case 0xC8:
    case 0xCA:
    case 0xCB:
    case 0xD8:
    case 0xDB:
    case 0xE8:
    case 0xEA:
    case 0xEB:
    case 0xF8:
    case 0xFB:
        pbuf[0] = 0;
        offset = 1;
        break;
        // Program Counter Relative
    case 0x10:
    case 0x30:
    case 0x50:
    case 0x70:
    case 0x80:
    case 0x90:
    case 0xB0:
    case 0xD0:
    case 0xF0:
        // Calculate the signed value of the param
        sval = (mem[1]>127) ? (mem[1]-256) : mem[1];
        sprintf(pbuf, "$%04lX", (pos+sval+2) & 0xFFFF);
        offset = 2;
        break;
        // Stack (Program Counter Relative Long)
    case 0x62:
        // Program Counter Relative Long
    case 0x82:
        // Calculate the signed value of the param
        sval = mem[1] + mem[2]*256;
        sval = (sval>32767) ? (sval-65536) : sval;
        sprintf(pbuf, "$%04lX", (pos+sval+3) & 0xFFFF);
        offset = 3;
        break;
        // Stack Relative Indirect Indexed, Y
    case 0x13:
    case 0x33:
    case 0x53:
    case 0x73:
    case 0x93:
    case 0xB3:
    case 0xD3:
    case 0xF3:
        sprintf(pbuf, "($%02X,S),Y", mem[1]);
        offset = 2;
        break;
        // Stack (Absolute)
    case 0xF4:
        sprintf(pbuf, "$%04X", mem[1] + mem[2]*256);
        offset = 3;
        break;
        // Stack (Direct Page Indirect)
    case 0xD4:
        sprintf(pbuf,"($%02X)",mem[1]);
        offset = 2;
        break;
        offset = 3;
        break;
        // Stack Relative
    case 0x03:
    case 0x23:
    case 0x43:
    case 0x63:
    case 0x83:
    case 0xA3:
    case 0xC3:
    case 0xE3:
        sprintf(pbuf,"$%02X,S",mem[1]);
        offset = 2;
        break;
        // WDM mode
    case 0x42:
        // Stack/Interrupt
    case 0x00:
    case 0x02:
        sprintf(pbuf,"$%02X",mem[1]);
        offset = 2;
        break;
        // Immediate (Invariant)
    case 0xC2:
        // REP following
        sprintf(pbuf,"#$%02X",mem[1]);
        offset = 2;
        break;
    case 0xE2:
        // SEP following
        sprintf(pbuf, "#$%02X", mem[1]);
        offset = 2;
        break;
        // Immediate (A size dependent)
    case 0x09:
    case 0x29:
    case 0x49:
    case 0x69:
    case 0x89:
    case 0xA9:
    case 0xC9:
    case 0xE9:
        if (flag & 0x20)
        {
            sprintf(pbuf, "#$%02X", mem[1]);
            offset = 2;
        }
        else
        {
            sprintf(pbuf,"#$%04X",mem[1]+mem[2]*256);
            offset = 3;
        }
        break;
        // Immediate (X/Y size dependent)
    case 0xA0:
    case 0xA2:
    case 0xC0:
    case 0xE0:
        if (flag & 0x10)
        {
            sprintf(pbuf,"#$%02X",mem[1]);
            offset = 2;
        }
        else
        {
            sprintf(pbuf,"#$%04X",mem[1]+mem[2]*256);
            offset = 3;
        }
        break;
    default:
        printf("Unhandled Addressing Mode: %02X\n",mem[0]);
        exit(1);
    };
    switch(mem[0]){
        // Absolute
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x1C:
    case 0x20:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x6D:
    case 0x6E:
    case 0x8C:
    case 0x8D:
    case 0x8E:
    case 0x9C:
    case 0xAC:
    case 0xAD:
    case 0xAE:
    case 0xCC:
    case 0xCD:
    case 0xCE:
    case 0xEC:
    case 0xED:
    case 0xEE:
        sprintf(pbuf,"$%04X",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indexed Indirect
    case 0x7C:
    case 0xFC:
        sprintf(pbuf,"($%04X,X)",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indexed, X
    case 0x1D:
    case 0x1E:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x5D:
    case 0x5E:
    case 0x7D:
    case 0x7E:
    case 0x9D:
    case 0x9E:
    case 0xBC:
    case 0xBD:
    case 0xDD:
    case 0xDE:
    case 0xFD:
    case 0xFE:
        sprintf(pbuf,"$%04X,X",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indexed, Y
    case 0x19:
    case 0x39:
    case 0x59:
    case 0x79:
    case 0x99:
    case 0xB9:
    case 0xBE:
    case 0xD9:
    case 0xF9:
        sprintf(pbuf,"$%04X,Y",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indirect
    case 0x6C:
        sprintf(pbuf,"($%04X)",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Indirect Long
    case 0xDC:
        sprintf(pbuf,"[$%04X]",mem[1]+mem[2]*256);
        offset=3;
        break;
        // Absolute Long
    case 0x0F:
    case 0x22:
    case 0x2F:
    case 0x4F:
    case 0x5C:
    case 0x6F:
    case 0x8F:
    case 0xAF:
    case 0xCF:
    case 0xEF:
        sprintf(pbuf,"$%06X",mem[1]+mem[2]*256+mem[3]*65536);
        offset=4;
        break;
        // Absolute Long Indexed, X
    case 0x1F:
    case 0x3F:
    case 0x5F:
    case 0x7F:
    case 0x9F:
    case 0xBF:
    case 0xDF:
    case 0xFF:
        sprintf(pbuf,"$%06X,X",mem[1]+mem[2]*256+mem[3]*65536);
        offset=4;
        break;
        // Accumulator
    case 0x0A:
    case 0x1A:
    case 0x2A:
    case 0x3A:
    case 0x4A:
    case 0x6A:
        sprintf(pbuf,"A");
        offset=1;
        break;
        // Block Move
    case 0x44:
    case 0x54:
        sprintf(pbuf,"$%02X,$%02X",mem[1],mem[2]);
        offset=3;
        break;
        // Direct Page
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x14:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x45:
    case 0x46:
    case 0x64:
    case 0x65:
    case 0x66:
    case 0x84:
    case 0x85:
    case 0x86:
    case 0xA4:
    case 0xA5:
    case 0xA6:
    case 0xC4:
    case 0xC5:
    case 0xC6:
    case 0xE4:
    case 0xE5:
    case 0xE6:
        sprintf(pbuf,"$%02X",mem[1]);
        offset=2;
        break;
        // Direct Page Indexed, X
    case 0x15:
    case 0x16:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x55:
    case 0x56:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x94:
    case 0x95:
    case 0xB4:
    case 0xB5:
    case 0xD5:
    case 0xD6:
    case 0xF5:
    case 0xF6:
        sprintf(pbuf,"$%02X,X",mem[1]);
        offset=2;
        break;
        // Direct Page Indexed, Y
    case 0x96:
    case 0xB6:
        sprintf(pbuf,"$%02X,Y",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect
    case 0x12:
    case 0x32:
    case 0x52:
    case 0x72:
    case 0x92:
    case 0xB2:
    case 0xD2:
    case 0xF2:
        sprintf(pbuf,"($%02X)",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect Long
    case 0x07:
    case 0x27:
    case 0x47:
    case 0x67:
    case 0x87:
    case 0xA7:
    case 0xC7:
    case 0xE7:
        sprintf(pbuf,"[$%02X]",mem[1]);
        offset=2;
        break;
        // Direct Page Indexed Indirect, X
    case 0x01:
    case 0x21:
    case 0x41:
    case 0x61:
    case 0x81:
    case 0xA1:
    case 0xC1:
    case 0xE1:
        sprintf(pbuf,"($%02X,X)",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect Indexed, Y
    case 0x11:
    case 0x31:
    case 0x51:
    case 0x71:
    case 0x91:
    case 0xB1:
    case 0xD1:
    case 0xF1:
        sprintf(pbuf,"($%02X),Y",mem[1]);
        offset=2;
        break;
        // Direct Page Indirect Long Indexed, Y
    case 0x17:
    case 0x37:
    case 0x57:
    case 0x77:
    case 0x97:
    case 0xB7:
    case 0xD7:
    case 0xF7:
        sprintf(pbuf,"[$%02X],Y",mem[1]);
        offset=2;
        break;
        // Stack (Pull)
    case 0x28:
    case 0x2B:
    case 0x68:
    case 0x7A:
    case 0xAB:
    case 0xFA:
        // Stack (Push)
    case 0x08:
    case 0x0B:
    case 0x48:
    case 0x4B:
    case 0x5A:
    case 0x8B:
    case 0xDA:
        // Stack (RTL)
    case 0x6B:
        // Stack (RTS)
    case 0x60:
        // Stack/RTI
    case 0x40:
        // Implied
    case 0x18:
    case 0x1B:
    case 0x38:
    case 0x3B:
    case 0x58:
    case 0x5B:
    case 0x78:
    case 0x7B:
    case 0x88:
    case 0x8A:
    case 0x98:
    case 0x9A:
    case 0x9B:
    case 0xA8:
    case 0xAA:
    case 0xB8:
    case 0xBA:
    case 0xBB:
    case 0xC8:
    case 0xCA:
    case 0xCB:
    case 0xD8:
    case 0xDB:
    case 0xE8:
    case 0xEA:
    case 0xEB:
    case 0xF8:
    case 0xFB:
        pbuf[0] = 0;
        offset = 1;
        break;
        // Program Counter Relative
    case 0x10:
    case 0x30:
    case 0x50:
    case 0x70:
    case 0x80:
    case 0x90:
    case 0xB0:
    case 0xD0:
    case 0xF0:
        // Calculate the signed value of the param
        sval = (mem[1]>127) ? (mem[1]-256) : mem[1];
        sprintf(pbuf, "$%04lX", (pos+sval+2) & 0xFFFF);
        offset = 2;
        break;
        // Stack (Program Counter Relative Long)
    case 0x62:
        // Program Counter Relative Long
    case 0x82:
        // Calculate the signed value of the param
        sval = mem[1] + mem[2]*256;
        sval = (sval>32767) ? (sval-65536) : sval;
        sprintf(pbuf, "$%04lX", (pos+sval+3) & 0xFFFF);
        offset = 3;
        break;
        // Stack Relative Indirect Indexed, Y
    case 0x13:
    case 0x33:
    case 0x53:
    case 0x73:
    case 0x93:
    case 0xB3:
    case 0xD3:
    case 0xF3:
        sprintf(pbuf, "($%02X,S),Y", mem[1]);
        offset = 2;
        break;
        // Stack (Absolute)
    case 0xF4:
        sprintf(pbuf, "$%04X", mem[1] + mem[2]*256);
        offset = 3;
        break;
        // Stack (Direct Page Indirect)
    case 0xD4:
        sprintf(pbuf,"($%02X)",mem[1]);
        offset = 2;
        break;
        offset = 3;
        break;
        // Stack Relative
    case 0x03:
    case 0x23:
    case 0x43:
    case 0x63:
    case 0x83:
    case 0xA3:
    case 0xC3:
    case 0xE3:
        sprintf(pbuf,"$%02X,S",mem[1]);
        offset = 2;
        break;
        // WDM mode
    case 0x42:
        // Stack/Interrupt
    case 0x00:
    case 0x02:
        sprintf(pbuf,"$%02X",mem[1]);
        offset = 2;
        break;
        // Immediate (Invariant)
    case 0xC2:
        // REP following
        sprintf(pbuf,"#$%02X",mem[1]);
        offset = 2;
        break;
    case 0xE2:
        // SEP following
        sprintf(pbuf, "#$%02X", mem[1]);
        offset = 2;
        break;
        // Immediate (A size dependent)
    case 0x09:
    case 0x29:
    case 0x49:
    case 0x69:
    case 0x89:
    case 0xA9:
    case 0xC9:
    case 0xE9:
        if (flag & 0x20)
        {
            sprintf(pbuf, "#$%02X", mem[1]);
            offset = 2;
        }
        else
        {
            sprintf(pbuf,"#$%04X",mem[1]+mem[2]*256);
            offset = 3;
        }
        break;
        // Immediate (X/Y size dependent)
    case 0xA0:
    case 0xA2:
    case 0xC0:
    case 0xE0:
        if (flag & 0x10)
        {
            sprintf(pbuf,"#$%02X",mem[1]);
            offset = 2;
        }
        else
        {
            sprintf(pbuf,"#$%04X",mem[1]+mem[2]*256);
            offset = 3;
        }
        break;
    default:
        printf("Unhandled Addressing Mode: %02X\n",mem[0]);
        exit(1);
    };


    // Generate whole disassembly line
    if(!(tsrc & 1))
    {
        // Generate hex output
        for (i = 0; i < offset; i++) {
            sprintf(hbuf + i * 2, "%02X", mem[i]);
        }
        for (i = offset * 2; i < 8; i++) {
            hbuf[i] = 0x20;
        }
        hbuf[8] = 0;
        sprintf(inst, "%02lX/%04lX:\t%s\t%s %s", (pos >> 16) & 0xFF, pos&0xFFFF, hbuf, ibuf, pbuf);
    }
    else
    {
        sprintf(inst, "%s %s", ibuf, pbuf);
    }

    return offset;
}
/* Pipe the output to :
 * | grep -B1 EXEC=NMI | grep -v EXEC=NMI | grep EXEC= | sort | uniq -c | sort -nr
 *
 */


unsigned int TOTAL_ITER = 0;
unsigned int MAX_ITER = 0;
unsigned int TOTAL_LOOPS = 0;


void write_visited(Word *shm, Word *visited, size_t count) {
	if (count > (VISITED_BUFFER_SIZE - 1)) {
		fprintf(stderr, "[in write_visited()] WARNING: count exceeds VISITED_BUFFER_SIZE: 0x%x\n", count);
		count = VISITED_BUFFER_SIZE - 1;
	}
	*shm = (Word) count;
	//int i;
	//for (i=0; i<count; i++) {
	//	*(shm + i + 1) = visited[i];
	//}
    memcpy((void *)(shm + 1), (void *) visited, count * sizeof(Word));
	return;
}

Word *init_tracing_shm(void) {
    char *retro_run_id;
	int rid = 1;
	if ((retro_run_id = getenv("RETRO_RUN_ID")))
    {
        rid = atoi(retro_run_id);
    };
	key_t key = (key_t) rid;
    int shmid = shmget(key, VISITED_BUFFER_SIZE * sizeof(Word), 0666|IPC_CREAT);
    Word *shm = (Word *) shmat(shmid, NULL, 0);
	if (shm == (void *) -1)
    {
        fprintf(stderr, "Failed to attach to shared memory! %d\n", errno);
        exit(errno);
    }
	memset((void *) shm, 0, (VISITED_BUFFER_SIZE * sizeof(Word)));
    return shm;
}

static inline void S9xReschedule (void);

#ifdef LAGFIX
bool8 finishedFrame = false;
#endif

void S9xMainLoop (void)
{
	//fprintf(stderr, "Entering main loop...\n");
	unsigned char *flag; // Used by the disassembler. TODO: integrate with emu
	Word *trace_shm = init_tracing_shm();
	size_t addr_count = 0;
	unsigned int iterations = 0;
	Word *visited;
	visited = (Word *) calloc(VISITED_BUFFER_SIZE, sizeof(Word));

	TOTAL_LOOPS++;

#ifdef LAGFIX
	do
	{
#endif
	for (;;)
	{
		iterations++;
		TOTAL_ITER++;
		if (CPU.NMILine)
		{
			if (Timings.NMITriggerPos <= CPU.Cycles)
			{
				CPU.NMILine = FALSE;
				Timings.NMITriggerPos = 0xffff;
				if (CPU.WaitingForInterrupt)
				{
					CPU.WaitingForInterrupt = FALSE;
					Registers.PCw++;
				}

				S9xOpcode_NMI();
#ifdef CPU_OPCODE_INSTRUMENTATION
            //puts("** EXEC=NMI");
#endif
			}
		}

		if (CPU.IRQTransition || CPU.IRQExternal)
		{
			if (CPU.IRQPending)
				CPU.IRQPending--;
			else
			{
				if (CPU.WaitingForInterrupt)
				{
					CPU.WaitingForInterrupt = FALSE;
					Registers.PCw++;
				}

				CPU.IRQTransition = FALSE;
				CPU.IRQPending = Timings.IRQPendCount;

				if (!CheckFlag(IRQ))
					S9xOpcode_IRQ();
			}
		}

//	#ifdef DEBUGGER
//		if ((CPU.Flags & BREAK_FLAG) && !(CPU.Flags & SINGLE_STEP_FLAG))
//		{
//			for (int Break = 0; Break != 6; Break++)
//			{
//				if (S9xBreakpoint[Break].Enabled &&
//					S9xBreakpoint[Break].Bank == Registers.PB &&
//					S9xBreakpoint[Break].Address == Registers.PCw)
//				{
//					if (S9xBreakpoint[Break].Enabled == 2)
//						S9xBreakpoint[Break].Enabled = TRUE;
//					else
//						CPU.Flags |= DEBUG_MODE_FLAG;
//				}
//			}
//		}
//
//		if (CPU.Flags & DEBUG_MODE_FLAG)
//			break;
//
//		if (CPU.Flags & TRACE_FLAG)
//			S9xTrace();
//
//		if (CPU.Flags & SINGLE_STEP_FLAG)
//		{
//			CPU.Flags &= ~SINGLE_STEP_FLAG;
//			CPU.Flags |= DEBUG_MODE_FLAG;
//		}
//	#endif

		if (CPU.Flags & SCAN_KEYS_FLAG) {
			break;
		}

		register uint8				Op;
		register struct	SOpcodes	*Opcodes;
		Word msg = 0;


        char *dis;
        dis = (char *) calloc(512, 1);
        uint64 inst_bytes = 0;
		uint8 flag = (uint8) Registers.P.W & 0xFF;
		Word offset = 0;

		if (CPU.PCBase)
		{
			offset = inst_offset(CPU.PCBase + Registers.PCw, Registers.P.W);
			// offset is always <= 4
			int i;
			for (i=offset-1;i>=0;i--) {
				inst_bytes <<= 8;
				inst_bytes |= CPU.PCBase[Registers.PCw + i];
			}

			Op = CPU.PCBase[Registers.PCw];
			CPU.PrevCycles = CPU.Cycles;
			CPU.Cycles += CPU.MemSpeed;
			S9xCheckInterrupts();
			Opcodes = ICPU.S9xOpcodes;
		}
		else
		{
			// NOTE: This section runs very infrequently
			Op = S9xGetByte(Registers.PBPC);
            inst_bytes |= Op;
			OpenBus = Op;
			Opcodes = S9xOpcodesSlow;
		}
        free(dis);

		// TODO: send compact representation of disassembled instructions over shm channel.

		if ((Registers.PCw & MEMMAP_MASK) + ICPU.S9xOpLengths[Op] >= MEMMAP_BLOCK_SIZE)
		{
			uint8	*oldPCBase = CPU.PCBase;

			CPU.PCBase = S9xGetBasePointer(ICPU.ShiftedPB + ((uint16) (Registers.PCw + 4)));
			if (oldPCBase != CPU.PCBase || (Registers.PCw & ~MEMMAP_MASK) == (0xffff & ~MEMMAP_MASK))
				Opcodes = S9xOpcodesSlow;
		}

        /******** Log the address visited. **************/
        Word old_pbpc = Registers.PBPC & 0xFFffff;
		//msg |= (uint64) flag << 24;
		/*** Message format:
		 * |leading zeroes?|bytecode (length <= 40 bits)|Block (8 bits)|Address (16 bits)|
		 *
		 */
		//if (visited[addr_count] == 0x2020) {
		//	fprintf(stderr, "[!] 0x%04x at visited[%d], Registers.PCw @ %p\n", visited[addr_count], addr_count, &(Registers.PCw));
		//}

		Registers.PCw++;
		(*Opcodes[Op].S9xOpcode)();

		if (!offset)
			offset = (Registers.PBPC - old_pbpc) & 0xFF;
		// TODO: Figure out how to get inst bytes for other case
		if (addr_count < VISITED_BUFFER_SIZE) {
            msg |= old_pbpc;
            msg |= offset << 24;
            msg |= inst_bytes << 32;
            visited[addr_count++] = (Word) msg;
        }
		//if (addr_count > 1 && (visited[addr_count-1] == visited[addr_count-2]))
	//		fprintf(stderr, "SUSPICIOUS REPETITION: %d: %X, %d: %X", addr_count-2, visited[addr_count-2], addr_count-1, visited[addr_count-1]);

		if (Settings.SA1)
			S9xSA1MainLoop();
			
#ifdef LAGFIX
		if (finishedFrame)
		break;
#endif
	}


#ifdef LAGFIX
	if (!finishedFrame)
#endif
	{
		S9xPackStatus();
	
		if (CPU.Flags & SCAN_KEYS_FLAG)
		{
			#ifdef DEBUGGER
				if (!(CPU.Flags & FRAME_ADVANCE_FLAG))
			#endif
				S9xSyncSpeed();
				CPU.Flags &= ~SCAN_KEYS_FLAG;
		}
	}
#ifdef LAGFIX
   else
   {
      finishedFrame = false;
      break;
   }
   }while(!finishedFrame);
#endif

    //fprintf(stderr, "Detaching from trace_shm\n");
    //fprintf(stderr, "Exiting main loop\n");
    //write_int(trace_shm, Registers.PCw);
	write_visited(trace_shm, visited, addr_count);
	addr_count = 0;
    shmdt(trace_shm);
	if (iterations > MAX_ITER) { MAX_ITER = iterations; }
    //fprintf(stderr, "Exiting main loop after %d iterations. (Mean: %d, Max: %d)\n", iterations, TOTAL_ITER / TOTAL_LOOPS, MAX_ITER);
	free(visited);
}

static inline void S9xReschedule (void)
{
	switch (CPU.WhichEvent)
	{
		case HC_HBLANK_START_EVENT:
			CPU.WhichEvent = HC_HDMA_START_EVENT;
			CPU.NextEvent  = Timings.HDMAStart;
			break;

		case HC_HDMA_START_EVENT:
			CPU.WhichEvent = HC_HCOUNTER_MAX_EVENT;
			CPU.NextEvent  = Timings.H_Max;
			break;

		case HC_HCOUNTER_MAX_EVENT:
			CPU.WhichEvent = HC_HDMA_INIT_EVENT;
			CPU.NextEvent  = Timings.HDMAInit;
			break;

		case HC_HDMA_INIT_EVENT:
			CPU.WhichEvent = HC_RENDER_EVENT;
			CPU.NextEvent  = Timings.RenderPos;
			break;

		case HC_RENDER_EVENT:
			CPU.WhichEvent = HC_WRAM_REFRESH_EVENT;
			CPU.NextEvent  = Timings.WRAMRefreshPos;
			break;

		case HC_WRAM_REFRESH_EVENT:
			CPU.WhichEvent = HC_HBLANK_START_EVENT;
			CPU.NextEvent  = Timings.HBlankStart;
			break;
	}
}

void S9xDoHEventProcessing (void)
{
#ifdef DEBUGGER
	static char	eventname[7][32] =
	{
		"",
		"HC_HBLANK_START_EVENT",
		"HC_HDMA_START_EVENT  ",
		"HC_HCOUNTER_MAX_EVENT",
		"HC_HDMA_INIT_EVENT   ",
		"HC_RENDER_EVENT      ",
		"HC_WRAM_REFRESH_EVENT"
	};
#endif

#ifdef DEBUGGER
	if (Settings.TraceHCEvent)
		S9xTraceFormattedMessage("--- HC event processing  (%s)  expected HC:%04d  executed HC:%04d",
			eventname[CPU.WhichEvent], CPU.NextEvent, CPU.Cycles);
#endif

	switch (CPU.WhichEvent)
	{
		case HC_HBLANK_START_EVENT:
			S9xReschedule();
			break;

		case HC_HDMA_START_EVENT:
			S9xReschedule();

			if (PPU.HDMA && CPU.V_Counter <= PPU.ScreenHeight)
			{
			#ifdef DEBUGGER
				S9xTraceFormattedMessage("*** HDMA Transfer HC:%04d, Channel:%02x", CPU.Cycles, PPU.HDMA);
			#endif
				PPU.HDMA = S9xDoHDMA(PPU.HDMA);
			}

			break;

		case HC_HCOUNTER_MAX_EVENT:
			if (Settings.SuperFX)
			{
				if (!SuperFX.oneLineDone)
					S9xSuperFXExec();
				SuperFX.oneLineDone = FALSE;
			}

			S9xAPUEndScanline();
			CPU.Cycles -= Timings.H_Max;
			CPU.PrevCycles -= Timings.H_Max;
			S9xAPUSetReferenceTime(CPU.Cycles);

			if ((Timings.NMITriggerPos != 0xffff) && (Timings.NMITriggerPos >= Timings.H_Max))
				Timings.NMITriggerPos -= Timings.H_Max;

			CPU.V_Counter++;
			if (CPU.V_Counter >= Timings.V_Max)	// V ranges from 0 to Timings.V_Max - 1
			{
				CPU.V_Counter = 0;
				Timings.InterlaceField ^= 1;

				// From byuu:
				// [NTSC]
				// interlace mode has 525 scanlines: 263 on the even frame, and 262 on the odd.
				// non-interlace mode has 524 scanlines: 262 scanlines on both even and odd frames.
				// [PAL] <PAL info is unverified on hardware>
				// interlace mode has 625 scanlines: 313 on the even frame, and 312 on the odd.
				// non-interlace mode has 624 scanlines: 312 scanlines on both even and odd frames.
				if (IPPU.Interlace && !Timings.InterlaceField)
					Timings.V_Max = Timings.V_Max_Master + 1;	// 263 (NTSC), 313?(PAL)
				else
					Timings.V_Max = Timings.V_Max_Master;		// 262 (NTSC), 312?(PAL)

				Memory.FillRAM[0x213F] ^= 0x80;
				PPU.RangeTimeOver = 0;

				// FIXME: reading $4210 will wait 2 cycles, then perform reading, then wait 4 more cycles.
				Memory.FillRAM[0x4210] = Model->_5A22;
				CPU.NMILine = FALSE;
				Timings.NMITriggerPos = 0xffff;

				ICPU.Frame++;
				PPU.HVBeamCounterLatched = 0;
				CPU.Flags |= SCAN_KEYS_FLAG;
			}

			// From byuu:
			// In non-interlace mode, there are 341 dots per scanline, and 262 scanlines per frame.
			// On odd frames, scanline 240 is one dot short.
			// In interlace mode, there are always 341 dots per scanline. Even frames have 263 scanlines,
			// and odd frames have 262 scanlines.
			// Interlace mode scanline 240 on odd frames is not missing a dot.
			if (CPU.V_Counter == 240 && !IPPU.Interlace && Timings.InterlaceField)	// V=240
				Timings.H_Max = Timings.H_Max_Master - ONE_DOT_CYCLE;	// HC=1360
			else
				Timings.H_Max = Timings.H_Max_Master;					// HC=1364

			if (Model->_5A22 == 2)
			{
				if (CPU.V_Counter != 240 || IPPU.Interlace || !Timings.InterlaceField)	// V=240
				{
					if (Timings.WRAMRefreshPos == SNES_WRAM_REFRESH_HC_v2 - ONE_DOT_CYCLE)	// HC=534
						Timings.WRAMRefreshPos = SNES_WRAM_REFRESH_HC_v2;					// HC=538
					else
						Timings.WRAMRefreshPos = SNES_WRAM_REFRESH_HC_v2 - ONE_DOT_CYCLE;	// HC=534
				}
			}
			else
				Timings.WRAMRefreshPos = SNES_WRAM_REFRESH_HC_v1;

			if (CPU.V_Counter == PPU.ScreenHeight + FIRST_VISIBLE_LINE)	// VBlank starts from V=225(240).
			{
				S9xEndScreenRefresh();

#ifdef LAGFIX
				if (!(GFX.DoInterlace && GFX.InterlaceFrame == 0)) /* MIBR */
                			finishedFrame = true;
#endif
				PPU.HDMA = 0;
				// Bits 7 and 6 of $4212 are computed when read in S9xGetPPU.
			#ifdef DEBUGGER
				missing.dma_this_frame = 0;
			#endif
				IPPU.MaxBrightness = PPU.Brightness;
				PPU.ForcedBlanking = (Memory.FillRAM[0x2100] >> 7) & 1;

				if (!PPU.ForcedBlanking)
				{
					PPU.OAMAddr = PPU.SavedOAMAddr;

					uint8	tmp = 0;

					if (PPU.OAMPriorityRotation)
						tmp = (PPU.OAMAddr & 0xFE) >> 1;
					if ((PPU.OAMFlip & 1) || PPU.FirstSprite != tmp)
					{
						PPU.FirstSprite = tmp;
						IPPU.OBJChanged = TRUE;
					}

					PPU.OAMFlip = 0;
				}

				// FIXME: writing to $4210 will wait 6 cycles.
				Memory.FillRAM[0x4210] = 0x80 | Model->_5A22;
				if (Memory.FillRAM[0x4200] & 0x80)
				{
					// FIXME: triggered at HC=6, checked just before the final CPU cycle,
					// then, when to call S9xOpcode_NMI()?
					CPU.NMILine = TRUE;
					Timings.NMITriggerPos = 6 + 6;
				}

			}

			if (CPU.V_Counter == PPU.ScreenHeight + 3)	// FIXME: not true
			{
				if (Memory.FillRAM[0x4200] & 1)
					S9xDoAutoJoypad();
			}

			if (CPU.V_Counter == FIRST_VISIBLE_LINE)	// V=1
				S9xStartScreenRefresh();

			S9xReschedule();

			break;

		case HC_HDMA_INIT_EVENT:
			S9xReschedule();

			if (CPU.V_Counter == 0)
			{
			#ifdef DEBUGGER
				S9xTraceFormattedMessage("*** HDMA Init     HC:%04d, Channel:%02x", CPU.Cycles, PPU.HDMA);
			#endif
				S9xStartHDMA();
			}

			break;

		case HC_RENDER_EVENT:
			if (CPU.V_Counter >= FIRST_VISIBLE_LINE && CPU.V_Counter <= PPU.ScreenHeight)
				RenderLine((uint8) (CPU.V_Counter - FIRST_VISIBLE_LINE));

			S9xReschedule();

			break;

		case HC_WRAM_REFRESH_EVENT:
		#ifdef DEBUGGER
			S9xTraceFormattedMessage("*** WRAM Refresh  HC:%04d", CPU.Cycles);
		#endif

			CPU.PrevCycles = CPU.Cycles;
			CPU.Cycles += SNES_WRAM_REFRESH_CYCLES;
			S9xCheckInterrupts();

			S9xReschedule();

			break;
	}

#ifdef DEBUGGER
	if (Settings.TraceHCEvent)
		S9xTraceFormattedMessage("--- HC event rescheduled (%s)  expected HC:%04d  current  HC:%04d",
			eventname[CPU.WhichEvent], CPU.NextEvent, CPU.Cycles);
#endif
}



