FROM ubuntu:latest
RUN apt-get update && apt-get -y install sudo
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker
COPY prerequisite.sh /tmp/prerequiste.sh
CMD /tmp/prerequiste.sh
