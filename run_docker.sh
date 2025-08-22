DOCKER_IMAGE=$1
NAME=$2
docker run -itd  --rm --privileged=true  --gpus all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
	-e NVIDIA_VISIBLE_DEVICES=all \
	--user $(id -u):$(id -g) \
	-p 25973:5901 \
	-p 22273:22 \
	--user 0 \
	--shm-size 256g \
	-v /data1/liy/projects/embodyai/Robotwin_Generation/embolab:/workspace/embolab \
	-v /data1/liy/projects/RoboTwin2.0:/workspace/robotwin_generation \
	--name $NAME \
	$DOCKER_IMAGE /bin/bash