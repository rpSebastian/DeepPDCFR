#!/bin/bash

. "${HOME}/miniconda3/etc/profile.d/conda.sh"
sudo apt-get install graphviz libgraphviz-dev gcc

conda create --name DeepPDCFR python=3.9 -y
conda activate DeepPDCFR

pip install -e .

