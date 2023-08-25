#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python finetune_hf_llm.py --model_path meta-llama/Llama-2-7b-chat \
                          --data_path marclove/llama_functions
