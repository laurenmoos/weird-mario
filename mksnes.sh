#! /bin/sh
make -j `nproc` -f CMakeFiles/clean-snes.dir/build.make CMakeFiles/clean-snes
make -j `nproc` -f CMakeFiles/snes.dir/build.make CMakeFiles/snes
cd Dispel
make
