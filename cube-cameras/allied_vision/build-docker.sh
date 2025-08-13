#!/bin/bash

docker build . -t camerafeed_allied:v1 \
		--build-arg UBUNTU_VERSION=20.04
