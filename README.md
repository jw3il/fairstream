# FairStream: Fair Multimedia Streaming Environment

[![arXiv](https://img.shields.io/badge/arXiv-2410.21029-b31b1b.svg)](https://arxiv.org/abs/2410.21029)

FairStream is an asynchronous multi-agent environment compatible with RLlib 2.10.

![](doc/scenario.svg)

In the environment, agents represents heterogeneous clients (see subfigure A) that stream on-demand multimedia content over a shared connection.
Given a time-varying bandwidth, the goal is to jointly optimize the Quality of Experience of each individual client and the Fairness across all clients.
The clients control the bitrate of their streams segment-based (see subfigure B) using partial observations and have heterogenous requirements (see perceptual quality in subfigure C), resulting in different reward functions for each client.

# Setup
The package can be installed using pip.
We recommend creating new python environment:

1. Create a conda/virtual environment (tested with Python 3.10.13)

    ```
    conda create -n fair-stream python=3.10
    ```

2. Install this project and its dependencies with pip (`-e` to keep the code editable):  

    ```
    (fair-stream) $ pip install .
    ``` 

The dependencies are listed in `pyproject.toml`.

The environment requires traces to run, which can be defined or downloaded separately (see section on network traces below).

## Getting Started

To confirm that the installation was successful, you can run an example that uses the environment with

```
python src/example.py
```

This will run a random agent in the environment (without requiring trace files) and print status information about each step.

## Network Traces
In our paper, we extracted traces from `curr_webget.csv` in the [FCC Raw Data Releases](https://www.fcc.gov/oet/mba/raw-data-releases) of the following dates: 2022 (July, August, November, December), 2023 (January, February, April, May, June, July). At the time of generating the dataset, the archive for March 2023 was corrupted and unusable. Exemplary scripts for downloading and extracting the required files can be found in [/datasets](/datasets/).

We generate our trace datasets and dataset-related plots with
```
python src/traces/cook.py
```
The datasets are saved in `./cooked_traces` and the plots in `./plots_traces`. Note that the trace dataset is split into training, validation, and test datasets.

## Training and Evaluation

Training and evaluation are performed with a single script `src/main.py`.
This script provides multiple options for training and evaluation that can be adjusting using the CLI. For a full explanation of all possible options run:
```
python src/main.py --help
```

It is the recommended start point for setting up your own experiments & environment & training loop.

The parameters that were used for our paper are provided in the [/scripts](/scripts) directory.

We provide a custom evaluation function, that allows tracking a variety of metrics per agent.
Beware that this more fine grained evaluation (data is split up into: trace_types, agents, metrics) comes with a more time consuming evaluation process.  

> [!CAUTION]
> If the custom evaluation is used, we **highly recommend**
>
> ```
> export TUNE_DISABLE_AUTO_CALLBACK_LOGGERS=1
> ```
> 
> before running the content to **prevent** the creation of **very big result files** by RLlib. The custom evaluation results will be available in experiment folder in the form of json files.

## Analysis and Plots
To monitor the training process, tensorboard can be used.
The output folder is typically set in the ray run_config of an experiment.
```
tensorboard --logdir=PATH_TO_RAY_OUTPUT_FOLDER
```

We also provide various scripts to create individual plots of the results in `src/plots/`. This also includes all scripts that have been used to create the plots in the paper. Note that the result directories in the scripts have to be adapted manually.
