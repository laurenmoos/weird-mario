config:
  environment.DISPLAY: :1
  nvidia.driver.capabilities: all
  nvidia.runtime: "true"
  user.user-data: |
    #cloud-config
    packages:
      - x11-apps
      - mesa-utils
description: GUI LXD profile
devices:
  X0:
    bind: container
    connect: unix:@/tmp/.X11-unix/X1/
    listen: unix:@/tmp/.X11-unix/X1/
    security.gid: "1000"
    security.uid: "1000"
    type: proxy
  mygpu:
    type: gpu
name: x11
used_by: []
