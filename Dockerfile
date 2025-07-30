ARG http_proxy
ARG https_proxy

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
RUN apt-get update && apt-get install -y \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    libegl1-mesa-dev libgl1-mesa-glx libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 \
    libglib2.0-0 libgtk-3-0 libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d
    
RUN apt-get update && apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-tools && rm -rf /var/lib/apt/lists/*
RUN /bin/bash -c "source /opt/conda/bin/activate && conda create -y -n RoboTwin python=3.10 && \
    echo 'source activate RoboTwin' >> ~/.bashrc"

# 安装Python依赖
RUN /bin/bash -c "source /opt/conda/bin/activate && source activate RoboTwin && \
    pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 \
    trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0"

#COPY . /workspace/robotwin_generation

#WORKDIR /workspace/robotwin_generation

# RUN /bin/bash -c "source /opt/conda/bin/activate && source activate RoboTwin && \
#     pip install /workspace/robotwin_generation/third_party/pytorch3d-0.7.8+pt2.4.0cu124-cp310-cp310-linux_x86_64.whl"


USER robotwin
