import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import glob
import os
import tqdm
import time

# Choose to use a config and initialize the detector
config = '../configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
config = '../configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
checkpoint = '../../../weights/mmdet/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth'
checkpoint = '../../../weights/mmdet/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth'

# Set the device to be used for evaluation
device = 'cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()
# Read images:
input_pth = "../../../t2i_out_samples/*"
input_imgs = glob.glob(os.path.expanduser(input_pth))
output_pth = "../../../insta_seg_out/"
assert input_imgs, "The input path(s) was not found"
for img_pth in tqdm.tqdm(input_imgs):
    result = inference_detector(model, img_pth)
    out_filename = os.path.join(output_pth, os.path.basename(img_pth))
    show_result_pyplot(model, img_pth, result, score_thr=0.3, out_file=out_filename)
