docker build --file ROCmDockerfile -t covidxpert_docker .
docker run --tty --interactive --publish 12000:8888 -v "/data2/covidxpert:/io/data" covidxpert_docker 