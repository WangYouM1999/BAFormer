#!/bin/bash

python ./train_supervision.py -c ./config/vaihingen/unetformer.py

bash vaihingen_test.sh

python ./train_supervision.py -c ./config/mapcup/unetformer.py

bash mapcup_test.sh

python ./train_supervision.py -c ./config/potsdam/unetformer.py

bash potsdam_test.sh