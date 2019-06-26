#!/bin/bash

# path of this script (code from stackoverflow)
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

nvidia-docker build -t tranx:latest $SCRIPTPATH/../docker
