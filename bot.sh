#!/bin/bash
python -m venv venv

source ./venv/bin/activate

pip install -r requirements.txt
pip install discord


python bot.py
