#!/usr/bin/env bash

if [ ! -d eigen ]; then
  echo Installing Eigen.
  wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 -P tmp/
  mkdir eigen
  tar xfj tmp/3.3.4.tar.bz2 -C eigen --strip-components=1
  echo Eigen installed.
else
  echo Eigen 3.3.4 already installed
fi
