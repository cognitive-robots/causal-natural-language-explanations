#!/bin/bash
export DOCKER_BUILDKIT=1
docker build --ssh default \
	-t causal_natural_language_explanations \
	.
