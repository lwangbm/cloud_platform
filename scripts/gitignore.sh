#!/usr/bin/env bash


SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/..
cd ${SCRIPT_ROOT}
echo "cd to ${SCRIPT_ROOT}"

find . -size +100M | cat >> .gitignore
#for file in $(find . -size +100M);
#do
#git rm --cached ${file}
#done


