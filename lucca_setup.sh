#! /usr/bin/env bash

# Quick and dirty setup for Lucca's PoC

pip install -e .
pip install sysv_ipc
pip install monkeyhex
./mksnes.sh
python -m retro.import ./SMW/

