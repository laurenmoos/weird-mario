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
