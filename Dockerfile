
FROM pytorch/pytorch
ENV USER=lauren
ENV HOME=/home/lauren

RUN mkdir -p $HOME
WORKDIR $HOME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 USER=$USER HOME=$HOME

RUN echo "The working directory is: $HOME"
RUN echo "The user is: $USER"

RUN apt-get update && apt-get -y install sudo
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY ./MarioWM/ $HOME/MarioWM

COPY ./RetroGym/ $HOME/RetroGym

RUN sudo apt-get update
RUN sudo apt-get -y install python3.8 python3.8-dev libbz2-dev pkg-config capnproto libcapnp-dev python3-opengl zlib1g-dev xvfb git build-essential cmake

RUN echo "Cloning Baselines"
WORKDIR $HOME
RUN git clone https://github.com/lejonet/baselines -b tf2
RUN pip3 install tensorflow
RUN pip3 install -e ./baselines/

WORKDIR $HOME
RUN echo "Installing lucca's fork of retrogym"
RUN pip3 install -e ./RetroGym/
WORKDIR $HOME/RetroGym
RUN python3 -m retro.import ./SMW


WORKDIR $HOME/MarioWM
RUN pip3 install toml
CMD ["main.py", "--exp-name", "--device"]
ENTRYPOINT ["python3"]
