docker run --gpus all -it --rm --name act-add -v $(pwd)/:/workspace -p :8888:8888 --entrypoint=/bin/bash drc/act-add:latest
