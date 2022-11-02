# Quirky_Quartet
This repository contains code for improving the performance of large language models (LLMs) in Code Generation tasks. This work was carried out as a semester project for the course CSE576: Natural Language Processing at Arizona State University in Fall 2022.  

### Installation 

To setup the repository, use `git clone --recurse-submodules -j8 git://github.com/skinahan/Quirky_Quartet`.  
If already downloaded, then update with `git submodule update --init --recursive`.

Install [Python 3.10](https://www.python.org/downloads/)  
Optionally install [poetry](https://python-poetry.org/) for package management.  
Optionally run `poetry update` to initialize and install all dependencies.  
_Alternatively_, you can use `poetry export -f requirements.txt --output requirements.txt` to install with conda, pip, or venv directly.  


### Sandboxing

To setup gvisor, first install [docker](https://docs.docker.com/engine/install/).
The gvisor install instructions are [here](https://gvisor.dev/docs/user_guide/install/), which has an option using `apt-get`.
Alternatively, run the gvisor install script: `./setup_gvisor`.
Then, setup with docker using:
``` bash
sudo /usr/local/bin/runsc install
sudo systemctl reload docker
docker run --rm --runtime=runsc hello-world
```

To setup an overlay filesystem (for safely using files within gvisor), edit `/etc/docker/daemon.json` to be:  
``` json
{
    "runtimes": {
        "runsc": {
            "path": "/usr/local/bin/runsc"
            "runtimeArgs": [
                    "--overlay"
            ]
        }
    }
}
```
When doing setup _only_, using the option `"--network=host"` could speed up the docker build, but it would be important to remove afterwards. 

Finally, build the docker container for this project:  
`docker build -t codex_codegen .`

Which was based on this [poetry docker setup](https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker).  

If the image breaks or needs to be rebuilt, you can use `docker image rmi codex_codegen`

Once the image is built successfully, the project can be run with:  
`docker run codex_codegen:latest`  
Which will execute the command specified by `CMD` in `Dockerfile`, by default `main.py`  

