#!/bin/zsh

# Managing Virtual Environment ###########################################

# Create a virtual environment if it doesn't exist
if [ ! -d venv ]; then
    python3 -m venv venv
    . ./venv/bin/activate
fi

# Activate the virtual environment
. ./venv/bin/activate
pip install -r requirements.txt

# Install non-pip packages ###########################################

######### NOTE: TRITON IS BROKEN ON MACOS #########
# export TRITON_DIR=$PWD/triton
# git clone https://github.com/openai/triton.git $TRITON_DIR
# cd $TRITON_DIR/python
# pip install cmake # build time dependency
# pip install -e .
# cd ../..
##########################################@@@@@@@@@