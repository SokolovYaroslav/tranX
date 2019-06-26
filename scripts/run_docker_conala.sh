#!/bin/bash

# path of this script (code from stackoverflow)
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

nvidia-docker run --rm -it \
    -v $SCRIPTPATH/..:/working_dir \
    tranx:latest bash -c "scripts/conala/train.sh 0"
