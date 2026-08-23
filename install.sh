#!/bin/bash

# Ensure pip is installed in the virtual enviroment
python3 -m ensurepip

# Install the project as a package
pip3 install -r requirements.txt

# Fix issue with numpy version
pip3 install --force-reinstall numpy==1.26.4
