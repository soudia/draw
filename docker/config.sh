export IMAGE_NAME="adware_docker"
export CONTAINER_NAME="adware"

# List of ports to map and publish
# Format: HOST_PORT_A:DOCKER_PORT_A HOST_PORT_B:DOCKER_PORT_B ...
export OPEN_PORTS="8888:8888"

# List of mounted volumes
# Format: HOST_DIR_A:DOCKER_DIR_A HOST_DIR_B:DOCKER_DIR_B ...
export MOUNTED_VOLUMES="/path/to/adware:/path/to/adware"

# Default working directory
export WORK_DIR="/adware"

# Specify GPU IDs
# Format: NV_GPU=0,1,3,5 or NV_GPU=0
export NV_HEADER="NV_GPU=0"
