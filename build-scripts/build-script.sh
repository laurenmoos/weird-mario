#!/usr/bin/env bash
err () {
  exit 1
}


trap err ERR

/usr/bin/env bash prerequisite.sh
/usr/bin/env bash retrogym-things.sh
