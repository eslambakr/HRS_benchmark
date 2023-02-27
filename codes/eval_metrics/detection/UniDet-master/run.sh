#!/bin/bash

/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/glide/vanilla_counting/*" --pkl_pth "../../counting/glide_pred_vanilla_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/glide/meta_counting/*" --pkl_pth "../../counting/glide_pred_meta_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/glide/synthetic_counting/*" --pkl_pth "../../counting/glide_pred_synthetic_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"

/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/minidalle/vanilla_counting/*" --pkl_pth "../../counting/minidalle_pred_vanilla_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/minidalle/meta_counting/*" --pkl_pth "../../counting/minidalle_pred_meta_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/minidalle/synthetic_counting/*" --pkl_pth "../../counting/minidalle_pred_synthetic_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"

/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/paella/vanilla_counting/*" --pkl_pth "../../counting/paella_pred_vanilla_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/paella/meta_counting/*" --pkl_pth "../../counting/paella_pred_meta_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/paella/synthetic_counting/*" --pkl_pth "../../counting/paella_pred_synthetic_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"

/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/sd_v1/synthetic_counting/*" --pkl_pth "../../counting/sd_v1_pred_synthetic_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"

/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/sd_v2/meta_counting/*" --pkl_pth "../../counting/sd_v2_pred_meta_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"
/home/eslam/anaconda3/envs/mmdet//bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/sd_v2/synthetic_counting/*" --pkl_pth "../../counting/sd_v2_pred_synthetic_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"