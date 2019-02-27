#!/bin/sh
source ~/.bashrc
python3 --version
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path=../res/sign_labels/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix ../res/sign_training/model.ckpt-7715 --output_directory ../res/sign_model
