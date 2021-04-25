docker build --file ROCmDockerfile -t covidxpert_docker .
docker run --device=/dev/kfd --device=/dev/dri \
    --group-add video --security-opt seccomp=unconfined \
    --tty --interactive --publish 12000:8888 -v "/data2/covidxpert:/io/data" covidxpert_docker
