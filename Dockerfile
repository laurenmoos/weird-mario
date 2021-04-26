
FROM pytorch/pytorch
ARG USER
ARG HOME

RUN mkdir -p $HOME
WORKDIR $HOME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 USER=$USER HOME=$HOME

RUN echo "The working directory is: $HOME"
RUN echo "The user is: $USER"

RUN apt-get update && apt-get -y install sudo
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY ./MarioWM/ /

COPY ./RetroGym/ /

RUN echo "Installing dependencies available from distro repositories"
#install tzdata package
RUN sudo apt -y install -y tzdata
# set your timezone
RUN sudo ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN sudo dpkg-reconfigure --frontend noninteractive tzdata
RUN sudo apt-get -y install -y build-essential cmake python3.8 python3.8-dev libbz2-dev pkg-config capnproto libcapnp-dev zlib1g-dev xvfb git curl python3-pip python3-opengl

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

RUN echo "Cloning Baselines"
WORKDIR $HOME
RUN git clone https://github.com/lejonet/baselines -b tf2
RUN pip3 install -e .

WORKDIR $HOME/RetroGym
RUN echo "Installing lucca's fork of retrogym"
RUN pip3 install -e .
RUN python3 -m retro.import ./SMW

WORKDIR $HOME/MarioWM
CMD ["main.py", "--exp-name", "--device"]
ENTRYPOINT ["python3"]