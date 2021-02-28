#!/usr/bin/env bash
err () {
  exit 1
}

trap err ERR

if [ $# -ne 1 ]; then
  echo "You need to supply a container name"
  echo "Usage:"
  echo "$0 <container name>"
  exit 1
else
  container_name="${1}"
fi

X11PROF=`mktemp`
cat>$X11PROF<<EOF
config:
  environment.DISPLAY: $DISPLAY
  environment.PULSE_SERVER: unix:/home/ubuntu/pulse-native
  nvidia.driver.capabilities: all
  nvidia.runtime: "true"
  user.user-data: |
    #cloud-config
    runcmd:
      - 'sed -i "s/; enable-shm = yes/enable-shm = no/g" /etc/pulse/client.conf'
    packages:
      - x11-apps
      - mesa-utils
      - alsa-utils
      - pulseaudio
      - paprefs
description: GUI LXD profile
devices:
  PASocket1:
    bind: container
    connect: unix:/run/user/1000/pulse/native
    listen: unix:/home/ubuntu/pulse-native
    security.gid: "1000"
    security.uid: "1000"
    uid: "1000"
    gid: "1000"
    mode: "0777"
    type: proxy
  X0:
    bind: container
    connect: unix:@/tmp/.X11-unix/X1
    listen: unix:@/tmp/.X11-unix/X0
    security.gid: "1000"
    security.uid: "1000"
    type: proxy
  mygpu:
    type: gpu
name: x11
used_by: []
EOF

echo "[+] Generated x11 profile in $X11PROF"
cat $X11PROF | lxc profile edit x11

lxc launch images:ubuntu/focal --profile default --profile x11 "${container_name}"
sleep 10
lxc file push ./build-script.sh "${container_name}/build-script.sh"
lxc exec "${container_name}" -- bash /build-script.sh
echo "[+] Stopping ${container_name} to make an image from it"
lxc stop "${container_name}"
lxc publish "${container_name}" --alias mario-dev-env
echo "[+] A new image based on ${container_name} can be launched/initialized with either"
echo "    lxc launch mario-dev-env [optional name] or lxc init mario-dev-env [optional name]"
echo
echo "[+] The container ${container_name} still exists and can also be used by just"
echo "    starting it with lxc start ${container_name}"
