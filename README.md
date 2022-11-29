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

### Agave Supercomputing Setup

To setup Python 3.10 on Agave, it needs to be compiled from scratch, which is less daunting than it seems. There is [good documentation](https://asurc.atlassian.net/wiki/spaces/RC/overview) on various aspects of Agave. In particulary, the [GPU](https://asurc.atlassian.net/wiki/spaces/RC/pages/45678646/Using+Graphics+Processing+Units+GPUs) documentation is helpful.  
Also, it is necessary to compile OpenSSL from scratch.
You can read more about why this is done/needed (and how) at [this](https://stackoverflow.com/questions/5937337/building-python-with-ssl-support-in-non-standard-location) stackoverflow post.  

First, login to the ASU VPN (or connect to ASU wifi), and then ssh into user@agave.asu.edu.
It is helpful to start an interactive session on a CPU cluster so that compiling is faster: `interactive -p htc -q normal -t 120`.  

First, load the gcc module using agave's software system (this may be necessary when doing `poetry update` as well):  
`module load gcc/11.2.0`  

Before installing Python, it is necessary to install OpenSSL 1.1.1, which Python 3.10 depends on. This needs to be done manually because admin access isn't available on Agave, and their OpenSSL version is 1.0.1:  
``` bash
mkdir openssl_111
wget https://www.openssl.org/source/openssl-1.1.1q.tar.gz
tar -xzvf openssl-1.1.1q.tar.gz
cd openssl-1.1.1q
./config --prefix=$HOME/openssl_111 --openssldir=$HOME/openssl_111/ssl
make -j 24
make test
make install
cd ..
```

Once the session is started, make a directory for the python executables, and then clone the cpython repository:
``` bash
mkdir python3.10
git clone https://github.com/python/cpython/
cd cpython
git checkout 3.10
```  
Importantly, the `./configure` step needs to enable optimizations and point to the directory we created:  
`CFLAGS="-I/home/username/openssl_111/include/" LDFLAGS="${LDFLAGS} -Wl,-rpath=/home/username/openssl_111/lib64" ./configure --prefix=$HOME/python3.10 --enable-optimizations --with-openssl=$HOME/openssl_111/`
After this is done, we can compile like normal, except that the install is done in our local directory.
``` bash
make -j 8
make -j 8 test
make install
```
If the tests fail, it is likely because of the OpenSSL library. You might see a message like "Could not build the ssl module!  Python requires a OpenSSL 1.1.1 or newer," in the log. If this is the case, check the linked stackoverflow about this, and follow the last answer (most recent). You can test that Python is successfully linked against OpenSSL by running `./python` and trying to `import ssl`.

To setup your path correctly, use these commands (potentially in `~/.bash_profile`):  
``` bash
export PATH=$HOME/python3.10/bin/:$PATH
alias python=python3.10
export PYTHONPATH=$HOME/python3.10
alias pip=pip3.10
```  
Then, install this package normally, starting with `pip install poetry`.  

To setup Cuda, use `module load cuda/11.6.0`.
To setup cudnn, it may be necessary to download it from [here](https://developer.nvidia.com/rdp/cudnn-download), `scp` it to agave, untar it, and then link to it:
``` bash
export INCLUDEPATH=$INCLUDEPATH:$HOME/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda/lib64
```

To setup git with tokens on Agave, obtain a token from GitHub, and use:
``` bash
git remote remove origin
git remote add origin https://[USERNAME]:[NEW TOKEN]@github.com/[USERNAME]/[REPO].git
```

