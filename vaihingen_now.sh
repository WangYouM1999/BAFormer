#!/bin/bash

python ./train_supervision.py -c ./config/vaihingen/unetformer.py

bash vaihingen_test.sh

