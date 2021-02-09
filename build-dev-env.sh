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

lxc launch images:ubuntu/focal "${container_name}"
sleep 10
lxc file push ./build-script.sh "${container_name}/build-script.sh"
lxc exec "${container_name}" -- bash /build-script.sh
