#!/bin/sh
set -Ceu

if [ $# -gt 0 ] && [ $1 == "clean" ]; then
  rm -f mzprojection.o mzprojection.mod example.o example.exe out*.dat
  exit
fi

gfortran -c mzprojection.f90
gfortran -c example.f90
gfortran mzprojection.o example.o -o example.exe
./example.exe
