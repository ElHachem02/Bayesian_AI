# Project 01

https://project.las.ethz.ch/task1/

## How to run
### Windows
docker build --tag task1 .; docker run --rm -u $(id -u):$(id -g) -v "$(pwd):/results" task1.