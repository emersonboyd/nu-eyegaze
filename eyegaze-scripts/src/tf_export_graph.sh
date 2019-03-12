#!/bin/sh
source ~/.bashrc
python3 --version
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path=../res/sign_labels/rfcn_resnet101_coco.config --trained_checkpoint_prefix ../res/sign_training/model.ckpt-14924 --output_directory ../res/sign_model
