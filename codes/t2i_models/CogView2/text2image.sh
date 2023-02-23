#!/bin/bash
# vim /home/abdelrem/anaconda3/envs/cog2/lib/python3.7/site-packages/SwissArmyTransformer/arguments.py

SAT_HOME=/home/abdelrem/t2i_benchmark/weights/t2i/cog2/pretrained
python cogview2_text2image.py \
       --mode inference \
       --fp16 \
       --input-source input.txt \
       --output-path "../../../data/t2i_out/cogview2/writing" \
       --batch-size 5 \
       --max-inference-batch-size 5 \
       --with-id \
       --device 0 \
       $@

