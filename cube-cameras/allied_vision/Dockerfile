ARG UBUNTU_VERSION=20.04

FROM vicoslab/echolib:${UBUNTU_VERSION} AS echolib

LABEL maintainer "pm4824@student.uni-lj.si"

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools
RUN python3 -m pip install opencv-python>=3 cython scipy

RUN mkdir /etc/udev
RUN mkdir /etc/udev/rules.d

# Install Allied Vision SDK
RUN cd /opt && wget https://downloads.alliedvision.com/Vimba_v5.0_Linux.tgz
RUN tar -xzf Vimba_v5.0_Linux.tgz && cd /opt/Vimba_5_0/VimbaUSBTL && ./Install.sh && cd /opt/Vimba_5_0/VimbaPython && ./Install.sh

ENV GENICAM_GENTL64_PATH=$GENICAM_GENTL64_PATH:"/opt/Vimba_5_0/VimbaUSBTL/CTI/x86_64bit"

COPY camera.py /opt/camera.py
COPY auto_calibration.py /opt/auto_calibration.py
COPY correction_dir/reference /opt/correction_dir/reference

RUN chmod +x /opt/camera.py

ENTRYPOINT ["/opt/camera.py"]
