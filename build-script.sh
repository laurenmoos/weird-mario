#!/usr/bin/env bash
err () {
  exit 1
}

trap err ERR

echo "Installing dependencies available from distro repositories"
apt update
apt install -y build-essential cmake python3.8 python3.8-dev libbz2-dev pkg-config capnproto libcapnp-dev zlib1g-dev xvfb python-opengl python3-pip git

echo "Installing dependencies with pip"
pip3 install torch tensorflow numpy~=1.19.2

echo "Cloning git repos"
git clone https://github.com/oblivia-simplex/retro
git clone https://github.com/openai/baselines -b tf2

echo "Installing lucca's fork of retrogym"
pushd ~/retro/RetroGym
pip3 install -e .
popd

echo "Installing a2c_ppo_acktr from MarioWM directory"
pushd ~/retro/MarioWM
pip3 install -e .
popd

echo "Installing baselines from own checkout"
pushd ~/baselines
pip3 install -e .
popd

