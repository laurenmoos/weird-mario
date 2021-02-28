#!/usr/bin/env bash
err () {
  exit 1
}

[ -d "retro-venv" ] || virtualenv retro-venv
source retro-venv/bin/activate

trap err ERR

echo "Installing dependencies available from distro repositories"
sudo apt update
sudo apt install -y build-essential cmake python3.8 python3.8-dev libbz2-dev pkg-config capnproto libcapnp-dev zlib1g-dev xvfb python-opengl python3-pip git

echo "Installing dependencies with pip"
pip3 install --pre torch
pip3 install tensorflow numpy~=1.19.2

echo "Cloning git repos"
git clone https://github.com/oblivia-simplex/retro
git clone https://github.com/lejonet/baselines -b tf2

echo "Installing lucca's fork of retrogym"
pushd retro
pushd RetroGym
pip3 install -e .
python3 -m retro.import ./SMW
popd

echo "Installing a2c_ppo_acktr from MarioWM directory"
pushd MarioWM
pip3 install -e .
popd
popd

echo "Installing baselines from own checkout"
pushd baselines
pip3 install -e .
popd
