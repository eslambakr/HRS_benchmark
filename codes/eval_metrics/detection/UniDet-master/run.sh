#!/bin/bash


/home/eslam/anaconda3/envs/mmdet/bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/struct_diff/synthetic_counting/*" --pkl_pth "../../counting/struct_diff_pred_synthetic_counting.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"

/home/eslam/anaconda3/envs/mmdet/bin/python -u demo.py --config-file configs/Partitioned_COI_RS101_2x.yaml \
--input "../../../../data/t2i_out/struct_diff/size/*" --pkl_pth "../../size_comp/struct_diff_pred_size.pkl" \
--output "../../../../data/metrics/det/unified_det" --opts MODEL.WEIGHTS "../../../../weights/unified_det/Partitioned_COI_RS101_2x.pth"