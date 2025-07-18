#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "prepare llama-factory for xtune"
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git fetch origin pull/8535/head:xtune && git checkout xtune && cd ..
rm -rf LLaMA-Factory/src/llamafactory/train/sft/workflow.py
rsync -avPr LLaMA-Factory/  .
rm -rf LLaMA-Factory
echo "prepare llama-factory done"
