#!/usr/bin/env bash
  err () {
    exit 1
  }
  trap err ERR

  pushd /tmp
  echo "Downloading and installing anaconda"
  curl https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh --output anaconda.sh
  chmod +x anaconda.sh
  ./anaconda.sh -b
  "${HOME}"/anaconda3/bin/conda init bash
  popd

  __conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
          . "/root/anaconda3/etc/profile.d/conda.sh"
      else
          export PATH="/root/anaconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  echo "Creating conda env 'retro-venv'"
  conda create --name retro-venv --yes python=3.8
  conda activate retro-venv
  echo "Installing dependencies with conda "
  conda install -y pytorch cudatoolkit=11.0 -c pytorch
  pip3 install pybullet
  pip3 install tensorflow
  pip3 install numpy==1.19.2
  echo "Cloning git repos"
  git clone https://github.com/laurenmoos/weird-mario
  git clone https://github.com/lejonet/baselines -b tf2
  echo "Installing lucca's fork of retrogym"
  pushd weird-mario
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
