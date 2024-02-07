#!/bin/bash

python run_a2_train.py  --oracle True ; python run_a2_train.py --oracle True --only_top True
python run_a2_train.py  --oracle False ; python run_a2_train.py --oracle False --only_top True
