#!/bin/bash

. "${HOME}/miniconda3/etc/profile.d/conda.sh"
sudo apt-get install graphviz libgraphviz-dev gcc

conda create --name IIG python=3.9 -y
conda activate IIG

cd ~/XDCFR
pip install -e .

cd ~/XHLib
pip install -e .
