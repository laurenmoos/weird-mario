#!/usr/bin/env bash

PROFILE="x11.lxd.profile"

lxc profile create x11

if [ ! $? -eq 0 ]; then
  echo "There seems to already be a x11 profile (see real error above)."
  echo "Do you want to remove the current x11 profile (you have to ensure that any container using it has been stopped)? [y/N]"
  read CHOICE
  if [ "${CHOICE}" == "y" ] || [ "${CHOICE}" == "Y" ]; then
    echo "Removing the x11 profile "
    lxc profile delete x11
    lxc profile create x11
  else
    echo "Exiting early to not clobber anything."
    exit 0
  fi
fi

echo "Configuring profile"
cat ${PROFILE} | lxc profile edit x11

echo "Done configuring profile, you can apply it at creation of a container by adding -p x11 to the lxc init or lxc launch"
echo "command or apply it to an already existing container with the following command: lxc profile add <instance name> x11"
echo
echo "If you don't have a nvidia card, you have to edit the profile (with command: lxc profile edit x11) and set"
echo "'nvidia.runtime: 'false' or else the container won't start at all."
echo "You can also set the runtime to false if you don't want to pass your nvidia card to the container."
echo
sed -i "s/environment\.DISPLAY: :0/environment.DISPLAY: $DISPLAY/" ${PROFILE}

echo "If the DISPLAY environment variable of the host doesn't output \"$DISPLAY\", you have to edit the"
echo "'connect: unix:@/tmp/.X11-unix/X0' line in the profile and replace X0 with X<the number> in the DISPLAY variable"

