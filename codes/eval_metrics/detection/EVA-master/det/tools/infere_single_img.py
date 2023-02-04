#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
"""
python tools/lazyconfig_train_net.py --num-gpus 1 \
    --eval-only \
    --config-file ../projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva_1536.py \
    "train.init_checkpoint=../../weights/eva_lvis.pth" \
    "dataloader.evaluator.max_dets_per_image=1000" \
    "model.roi_heads.maskness_thresh=0.5" # use maskness to calibrate mask predictions
"""
import logging
import glob
import os
import tqdm
import time
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def run_on_image(image, model):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        predictions (dict): the output of the model.
        vis_output (VisImage): the visualized image output.
    """
    vis_output = None
    cpu_device = torch.device("cpu")
    instance_mode = ColorMode.IMAGE
    metadata = MetadataCatalog.get("__unused")
    model.training = False
    predictions = model([{"image": torch.as_tensor(image.copy()).permute(2, 0, 1)}])
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    image = image[:, :, ::-1]
    visualizer = Visualizer(image, metadata, instance_mode=instance_mode)
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to(cpu_device), segments_info
        )
        print("Vis is --> panoptic_seg")
    else:
        if "sem_seg" in predictions:
            vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(cpu_device)
            )
            print("Vis is --> sem_seg")
        if "instances" in predictions:
            instances = predictions["instances"].to(cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            print("Vis is --> instances")

    return predictions, vis_output


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        model.training = False
        model.proposal_generator.training = False
        model.roi_heads.training = False
        model.backbone.training = False
        model.roi_heads.mask_head.training = False
        model.roi_heads.box_head.training = False
        model.roi_heads.box_predictor.training = False
        model.roi_heads.mask_pooler.training = False

        # Read images:
        input_pth = "../../../../t2i_out_samples/*"
        input_imgs = glob.glob(os.path.expanduser(input_pth))
        output_pth = "../out_samples/"
        assert input_imgs, "The input path(s) was not found"
        for path in tqdm.tqdm(input_imgs, disable=not output_pth):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = run_on_image(img, model)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if output_pth:
                if os.path.isdir(output_pth):
                    assert os.path.isdir(output_pth), output_pth
                    out_filename = os.path.join(output_pth, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
