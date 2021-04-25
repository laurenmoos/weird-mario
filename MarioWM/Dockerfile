FROM ubuntu:latest

RUN apt-get update && apt-get -y install sudo
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker
#TODO: this will persist after build and is not a best practice
#COPY prerequisite.sh /tmp/prerequiste.sh
#CMD /tmp/prerequiste.sh
RUN echo "Installing dependencies available from distro repositories"
#install tzdata package
RUN sudo apt -y install -y tzdata
# set your timezone
RUN sudo ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN sudo dpkg-reconfigure --frontend noninteractive tzdata
RUN sudo apt-get -y install -y build-essential cmake python3.8 python3.8-dev libbz2-dev pkg-config capnproto libcapnp-dev zlib1g-dev xvfb git curl python3-pip python3-opengl

RUN echo 'initial directory'
RUN pwd
COPY prerequisite.sh .
RUN sudo ./prerequisite.sh
RUN echo 'directory after run'
RUN pwd

#COPY /weird-mario/MarioWM $HOME/MarioWM
WORKDIR $HOME/MarioWM
#CMD ["MarioWM/main.py", "--exp-name", "--device"]
#ENTRYPOINT ["python3"]