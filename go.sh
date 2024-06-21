#!/bin/bash

python main.py \
    --model "configs/model/Model_1.py" \
    --data "configs/data/data_params_1.py" \
    --features "configs/features/features_5.py"

python main.py \
    --model "configs/model/Model_2.py" \
    --data "configs/data/data_params_1.py" \
    --features "configs/features/features_5.py"

python main.py \
    --model "configs/model/Model_3.py" \
    --data "configs/data/data_params_1.py" \
    --features "configs/features/features_5.py"

python main.py \
    --model "configs/model/Model_1.py" \
    --data "configs/data/data_params_2.py" \
    --features "configs/features/features_5.py"

python main.py \
    --model "configs/model/Model_2.py" \
    --data "configs/data/data_params_2.py" \
    --features "configs/features/features_5.py"

python main.py \
    --model "configs/model/Model_3.py" \
    --data "configs/data/data_params_2.py" \
    --features "configs/features/features_5.py"