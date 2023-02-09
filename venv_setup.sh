#!/bin/zsh

# Create a virtual environment if it doesn't exist
if [ ! -d venv ]; then
    python3 -m venv venv
    . ./venv/bin/activate
    pip install -r requirements.txt
fi

# Activate the virtual environment
. ./venv/bin/activate