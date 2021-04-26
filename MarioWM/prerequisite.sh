#!/usr/bin/env bash
#  err () {
#    exit 1
#  }
#  trap err ERR
#
#  echo "Downloading and installing anaconda"
#  curl https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh --output anaconda.sh
#  chmod +x anaconda.sh
#  ./anaconda.sh -b
#  "${HOME}"/anaconda3/bin/conda init bash
#  # {/tmp}
#
#  __conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#  if [ $? -eq 0 ]; then
#      eval "$__conda_setup"
#  else
#      if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
#          . "/root/anaconda3/etc/profile.d/conda.sh"
#      else
#          export PATH="/root/anaconda3/bin:$PATH"
#      fi
#  fi
#  unset __conda_setup
  RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh \
    && /bin/bash ~/anaconda.sh -b -p /opt/conda \
    && rm ~/anaconda.sh

  echo "Creating conda env 'retro-venv'"
  conda create --name retro-venv --yes python=3.8
  conda activate retro-venv
  echo "Installing dependencies with conda "
  conda install -y pytorch cudatoolkit=11.0 -c pytorch
  pip3 install pybullet
  pip3 install numpy==1.19.2

