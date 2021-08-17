#!/usr/bin/env bash

# project root
cd "$(dirname "${BASH_SOURCE[0]}")"/../

export PYTHONPATH="/home/luping/work/"
#export PYTHONPATH="/Users/ourokutaira/Desktop/InfoCom/Code/:$PYTHONPATH"



if [[ "$DEBUG" == "TRUE" ]]; then
    exec ipython -m pdb master/master.py SEBF dummy 0.8
else
    exec python master/master.py "$@"
fi
