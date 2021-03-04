# Introduction

The build script consists of two parts, the "outside" build-script (called build-dev-env.sh) and the "inside" build-scripts (in the tarball build-scripts.tar.gz).

Executing ./build-dev-env.sh <container name> will make the "outside" script start a container name <container name>, wait for the containers network to settle, push the "inside" script tarball to the container, extract the tarball and execute the "inside" build script to actually install the dependencies and so.

Besides the actual build-scripts, there is a lxd-config.sh script that will create a x11 profile (based off the x11.lxd.profile file) and output some instructions on how you can use the profile.

# LXD stuff

To bind mount a folder from the host into the container, for example to gather logs from the container, you can do the following:
* Ensure that the bind-mount is in one of the profiles that the container has, an example profile to bind /path/to/logs from the host to /logs in the container would be the following:
```
config: {}
description: ""
devices:
  logs:
    path: /logs
    source: /path/to/logs
    type: disk
name: logs-directory
```

* It can also be done on a ad-hoc basis with the following command: `lxc config device add <container name> <mount tag, can be anything> disk source=/path/to/logs path=/logs`

As the image is fairly large by itself (currently around 13-14GB), LXD will most likely create a container with a rootfs that is just big enough to fit the image and not much else. This means that unless you tell LXD to create a rootfs with a larger size, you'll very quickly run into "out of space" errors.
The easiest way to ensure that doesn't happen is to add a size parameter to the root disk defined in the default profile. It would look like the following:
```
config: {}
description: Default LXD profile
devices:
  eth0:
    name: eth0
    network: lxdbr0
    type: nic
  root:
    path: /
    pool: default
    type: disk
    size: 50GB
used_by: []
```

# The container env

Inside of the container, everything is installed into a conda environment called retro-venv, so the first thing that has to be done after entering the container is `conda activate retro-venv` to get access to the installed dependencies and be able to run the code.
