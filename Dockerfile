FROM python:3.10.8-alpine3.16

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.13

# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry update -vvv --no-interaction 
RUN poetry install -vvv --no-interaction 

# Creating folders, and files for a project:
COPY . /code
CMD ls && cd /code && ls && poetry run python main.py hello_world
