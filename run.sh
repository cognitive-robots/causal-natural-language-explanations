#!/bin/bash

# get input args
docker_run_args="$*"

# mount volumes
docker_run_cmd="docker run --rm \
		--volume=$HOME/.Xauthority:/root/.Xauthority:rw \
		-v /Volumes/:/Volumes/:rw \
		-v $HOME/Workspace/causal_natural_language_explanations:/root/Workspace/causal_natural_language_explanations:rw
		--env='DISPLAY'"

# optionally expose tensorboard mount
if [[ $docker_run_args = *"with_tb"* ]]; then
	docker_run_cmd="${docker_run_cmd} -p 5118:5118"
fi

# optionally mount all code
if [[ $docker_run_args = *"load_code"* ]]; then
        docker_run_cmd="${docker_run_cmd} -v $HOME/code/stash/sax/sax-logs:/root/code/stash/sax/sax-logs:rw"
fi

# optionally use host GPUs
if [[ $docker_run_args = *"with_gpu"* ]]; then
	#docker_run_cmd="${docker_run_cmd} --gpus all"
	docker_run_cmd="${docker_run_cmd} --runtime=nvidia"
fi

# latest tag
docker_run_cmd="${docker_run_cmd} --ipc=host \
		-it \
		causal_natural_language_explanations"

# run
$docker_run_cmd
