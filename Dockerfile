# SLAM robot + object detector: ROS 2 Humble, Gazebo Classic, Nav2, YOLO26 on the CPU.
# Libraries for ROS, Gazebo, Nav2, RViz and the display libraries.

FROM ros:humble-ros-base-jammy AS base

ARG DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-turtlebot3-gazebo \
        ros-humble-nav2-bringup \
        ros-humble-slam-toolbox \
        ros-humble-rviz2 \
        ros-humble-rqt-image-view \
        python3-venv \
        libgl1 libglib2.0-0 libgl1-mesa-dri libglx-mesa0 libegl1 mesa-utils \
        x11-apps fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python environment for the detector (PyTorch, Ultralytics, OpenCV).
FROM base AS python

ENV SLAM_VENV=/opt/venv \
    PYTHONNOUSERSITE=1

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv --system-site-packages "$SLAM_VENV" \
 && "$SLAM_VENV/bin/pip" install --no-cache-dir --upgrade pip \
 && "$SLAM_VENV/bin/pip" install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# My ROS package, built.
FROM python AS workspace

# ROS setup.bash needs bash; the default sh cannot read it
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /ws
COPY slam_robot src/slam_robot

# Anything package.xml needs that stage 1 does not cover. It installs nothing as long as stage 1 is complete.
RUN apt-get update \
 && rosdep install --from-paths src --ignore-src -r -y \
        --dependency-types=build --dependency-types=buildtool --dependency-types=exec \
 && rm -rf /var/lib/apt/lists/*

# Built with the VENV's Python (python -m colcon), so the start script of detector_node runs in the venv,
# where ultralytics is.
# hadolint ignore=SC1091
RUN . /opt/ros/humble/setup.bash \
 && "$SLAM_VENV/bin/python" -m colcon build \
 && rm -rf build log

# Stop the build here, not at the first launch, if the detector would start with the wrong Python.
RUN test "$(head -n 1 install/slam_robot/lib/slam_robot/detector_node)" = "#!$SLAM_VENV/bin/python"

# Trained weights and the 'slam' command. This is the image you run.

FROM workspace AS runtime

# Links the image to this repository on GitHub (GitHub Container Registry reads the 'source' label) and states the licence.
LABEL org.opencontainers.image.title="SLAM robot object search" \
      org.opencontainers.image.description="Gazebo + Nav2 + YOLO26 object search robot (ROS 2 Humble), with the trained detector" \
      org.opencontainers.image.source="https://github.com/Tausif30/SLAM_Robot" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

COPY --chmod=0755 docker/slam /usr/local/bin/slam
COPY weights/detector_v1.pt /opt/slam/models/detector_v1.pt

# SLAM_MODEL_PATH: where the detector finds its weights
# GAZEBO_MODEL_DATABASE_URI="": do not wait for the online model database (it hangs without internet)
# LIBGL_ALWAYS_SOFTWARE: draw with the CPU, works on every machine
# QT_X11_NO_MITSHM: Qt windows through the X11 socket of another namespace
ENV SLAM_MODEL_PATH=/opt/slam/models/detector_v1.pt \
    GAZEBO_MODEL_DATABASE_URI="" \
    LIBGL_ALWAYS_SOFTWARE=1 \
    QT_X11_NO_MITSHM=1

# A settings folder for Ultralytics, and ROS in every interactive shell.
RUN mkdir -p /root/.config/Ultralytics \
 && echo '. /opt/ros/humble/setup.bash; . /ws/install/setup.bash' >> /root/.bashrc

ENTRYPOINT ["/usr/local/bin/slam"]
CMD ["shell"]
