#!/usr/bin/env bash
err () {
  exit 1
}

trap err ERR

echo "Creating conda env 'retro-venv'"
conda create --name retro-venv --yes python=3.8
conda activate retro-venv

echo "Installing dependencies with conda"
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install tensorflow numpy~=1.19.2

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

