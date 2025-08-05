FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}


RUN /bin/bash -c "source /opt/conda/bin/activate && conda create -y -n RoboTwin python=3.10 && \
    echo 'source activate RoboTwin' >> ~/.bashrc"

RUN apt-get update && apt-get install -y \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    libegl1-mesa-dev libgl1-mesa-glx libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 \
    libglib2.0-0 libgtk-3-0 libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d

# 安装Python依赖
RUN /bin/bash -c "source /opt/conda/bin/activate && source activate RoboTwin && \
    pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 \
    trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0"

RUN /bin/bash -c "source /opt/conda/bin/activate && source activate RoboTwin && \
    pip install zarr==2.12.0 wandb ipdb gpustat dm_control \
    omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 \
    diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib \
    termcolor" 

