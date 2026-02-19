FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

# 基本ツール + USB + ROS関連(TF/Control/MoveIt)
RUN apt-get update && apt-get install -y \
    usbutils \
    udev \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-pykdl \
    python3-rosdep \
    python3-catkin-pkg \
    python3-rosinstall \
    # TF関連
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-tf-transformations \
    # 診断・制御・MoveIt関連 (追加分)
    ros-humble-diagnostic-aggregator \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ros2controlcli \
    && rm -rf /var/lib/apt/lists/*

# pip パッケージ
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      pyserial \
      pybullet \
      transforms3d

# rosdep 初期化
RUN rosdep init || true
RUN rosdep update || true

# ワークスペース用ディレクトリ
WORKDIR /ros2_ws

# デフォルトで bash
CMD ["bash"]