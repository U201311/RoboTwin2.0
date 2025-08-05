FROM data_generation_v2:v2 

ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}



RUN /bin/bash -c "source /opt/conda/bin/activate && source activate RoboTwin && \
    pip install zarr==2.12.0 wandb ipdb gpustat dm_control \
    omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 \
    diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib \
    termcolor" 

