#!/bin/bash -e

docker build -f Dockerfile_tests -t test-pyutils .
docker run --rm -it test-pyutils tests
