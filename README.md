# Quirky_Quartet
This repository contains code for improving the performance of large language models (LLMs) in Code Generation tasks. This work was carried out as a semester project for the course CSE576: Natural Language Processing at Arizona State University in Fall 2022.  

### Installation 

To setup the repository, use `git clone --recurse-submodules -j8 git://github.com/skinahan/Quirky_Quartet`.  
If already downloaded, then update with `git submodule update --init --recursive`.

Install [Python 3.10](https://www.python.org/downloads/)  
Optionally install [poetry](https://python-poetry.org/) for package management.  
Optionally run `poetry update` to initialize and install all dependencies.  
_Alternatively_, you can use `poetry export -f requirements.txt --output requirements.txt` to install with conda, pip, or venv directly.  

### Usage

The main entrypoint is the `./run` bash script. By default, this runs the _script_ `scripts/main.py`, or if provided with an argument, such as `./run download`, then it runs the corresponding script, e.x. `scripts/download.py`.

