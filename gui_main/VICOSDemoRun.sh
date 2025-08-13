#!/usr/bin/env bash

# set VICOS_CUBE_GUI_DIR to the directory of this script
export VICOS_CUBE_GUI_DIR="$(dirname "$(readlink -f "$0")")"
export VICOS_CUBE_VIRTENV=/home/vicosdemo/demo

source $VICOS_CUBE_VIRTENV/bin/activate

xhost +local:docker

# start DEMO using ignition
python -m ignition $VICOS_CUBE_GUI_DIR/launch_demo.yaml

# Clean up: stop and then remove any unclosed containers
echo ""
echo "Cleaning up any remaining containers"
docker stop $(docker ps -q) &> /dev/null
docker rm $(docker ps -a -q) &> /dev/null
echo "done"
