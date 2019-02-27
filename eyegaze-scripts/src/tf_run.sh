#!/bin/sh
source ~/.bashrc
python3 --version
python3 train.py --logtostderr --train_dir=../res/sign_training --pipeline_config_path=../res/sign_labels/ssd_mobilenet_v1_pets.config

