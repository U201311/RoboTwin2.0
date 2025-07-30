DOCKER_IMAGE=$1
NAME=$2
docker run -itd  --rm --privileged=true  --gpus all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
	-e NVIDIA_VISIBLE_DEVICES=all \
	--user $(id -u):$(id -g) \
	-p 25913:5901 \
	-p 22213:22 \
	--user 0 \
	--shm-size 128g \
	-v /data1/liy/projects/RoboTwin2.0/para/build_task.json:/workspace/robotwin_generation/para/build_task.json \
    -v /data1/liy/projects/embodyai/Robotwin_Generation/data:/workspace/robotwin_generation/data \
    -v /data1/liy/projects/embodyai/Robotwin_Generation/logs:/workspace/robotwin_generation/logs \
	--name $NAME \
	$DOCKER_IMAGE /bin/bash