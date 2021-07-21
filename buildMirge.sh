#!/bin/bash

usage()
{
  echo "Usage: $0 [options]"
  echo "  --use-ssh         Use ssh-keys to clone emirge/mirgecom"
  echo "  --help            Print this help text."
}

opt_git_ssh=0

while [[ $# -gt 0 ]];do
  arg=$1
  shift
  case $arg in
    --use-ssh)
      opt_git_ssh=1
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "=== Error: unknown argument '$arg' ."
      usage
      exit 1
      ;;
  esac
done

# default install script for mirgecom, will build with the current development head
echo "Building MIRGE-Com from the current development head"
echo "***WARNING*** may not be compatible with this driver ***WARNING"
echo "consider build scripts from <platforms> as appropriate"

if [ -z "$(ls -A emirge)" ]; then
  if [ $opt_git_ssh -eq 1 ]; then
    echo "git clone git@github.com:illinois-ceesd/emirge.git emirge"
    git clone git@github.com:illinois-ceesd/emirge.git emirge
  else
    echo "git clone https://github.com/illinois-ceesd/emirge.git emirge"
    git clone https://github.com/illinois-ceesd/emirge.git emirge
  fi
else
  echo "emirge install already present. Remove to build anew"
fi

cd emirge
git_method=""
if [ $opt_git_ssh -eq 1 ]; then
  git_method="--git-ssh"
fi

if [ -z ${CONDA_PATH+x} ]; then
  echo "CONDA_PATH unset, installing new conda with emirge"
  echo "./install.sh --env-name=mirgeDriver.Y1nozzle $git_method --branch=y1-production"
  ./install.sh --env-name=mirgeDriver.Y1flatPlate $git_method --branch=y1-production
else
  echo "Using existing Conda installation, ${CONDA_PATH}"
  echo "./install.sh --conda-prefix=$CONDA_PATH --env-name=mirgeDriver.Y1nozzle $git_method --branch=y1-production"
  ./install.sh --conda-prefix=$CONDA_PATH --env-name=mirgeDriver.Y1flatPlate $git_method --branch=y1-production
fi
