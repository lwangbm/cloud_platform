#!/usr/bin/env bash

# project root
cd "$(dirname "${BASH_SOURCE[0]}")"/../

export PYTHONPATH="/home/luping/work/:$PYTHONPATH"


if [[ "$DEBUG" == "TRUE" ]]; then
    exec ipython -m pdb slave/slave.py
else
    if [[ "$1" == "production" ]]; then
        exec python slave/slave.py &> logs
    else
        exec python slave/slave.py
    fi
fi
