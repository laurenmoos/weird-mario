#!/usr/bin/env bash
err () {
  exit 1
}

trap err ERR

echo "Installing dependencies available from distro repositories"
sudo apt update
sudo apt install -y build-essential cmake python3.8 python3.8-dev libbz2-dev pkg-config capnproto libcapnp-dev zlib1g-dev xvfb git curl python3-pip python3-opengl

pushd /tmp
echo "Downloading and installing anaconda"
curl https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh --output anaconda.sh
chmod +x anaconda.sh
./anaconda.sh -b
"${HOME}"/anaconda3/bin/conda init bash
popd

