#!/bin/bash

python main.py \
    --model "configs/model/Model_1.py" \
    --data "configs/data/data_params.py" \
    --features "configs/features/features_2.py"

python main.py \
    --model "configs/model/Model_2.py" \
    --data "configs/data/data_params.py" \
    --features "configs/features/features_2.py"

python main.py \
    --model "configs/model/Model_3.py" \
    --data "configs/data/data_params.py" \
    --features "configs/features/features_2.py"