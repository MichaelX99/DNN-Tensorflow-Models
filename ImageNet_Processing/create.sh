#!/bin/bash

for entry in `ls -1 *.tar`; do
    mkdir ${entry:0:9}
    mv $entry ${entry:0:9}
    cd ${entry:0:9}
    tar -xf $entry
    rm $entry
    cd ..
    echo ${entry:0:9}
done
