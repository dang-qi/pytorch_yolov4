import torch
import cv2
import numpy as np
import os
import time
import tqdm
import json
from torchvision.ops import nms
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator
from tool import utils

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils.data import DataLoader
from models import Yolov4
from tool.darknet2pytorch import Darknet

from cfg_patch import Cfg
from dataset import YoloModanetHumanDataset
from tool.tv_reference.utils import collate_fn as val_collate

@torch.no_grad()
def evaluate_nms(model, data_loader, cfg, device, human_patch=False, json_gt=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    if json_gt is None:
        coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
        coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')
    else:
        coco_gt = COCO(json_gt)

    result_json = []
    for images, targets in tqdm.tqdm(data_loader, desc='testing'):
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)
        outputs = utils.post_processing(conf_thresh=0.001, nms_thresh=0.5, output=outputs)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, output in zip(images, targets, outputs):
            img_height, img_width = img.shape[:2]
            #human_box = target['human_box']
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = output[:,:4]
            scores = output[:,-2]
            labels = output[:,-1]

            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            if human_patch:
                human_box = target['human_box'].cpu().detach().numpy()
                boxes[...,0] = boxes[...,0]*img_width + human_box[0]
                boxes[...,1] = boxes[...,1]*img_height + human_box[1]
            else:
                boxes[...,0] = boxes[...,0]*img_width 
                boxes[...,1] = boxes[...,1]*img_height 
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height

            if json_gt is None:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # confs = output[...,4:].copy()
                labels = torch.as_tensor(labels, dtype=torch.int64)
                scores = torch.as_tensor(scores, dtype=torch.float32)
                res[target["image_id"].item()] = {
                    "boxes": boxes.unsqueeze(1),
                    "scores": scores,
                    "labels": labels,
                }
            else:
                for box, label, score in zip(boxes, labels, scores):
                    single_result = {'image_id': int(target['image_id'].item()),
                                    'bbox': box.tolist(),
                                    'category_id': int(label+1),
                                    'score':float(score)}
                    result_json.append(single_result)
        evaluator_time = time.time()
        if json_gt is None:
            coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    if json_gt is None:
        # gather the stats from all processes
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator
    else:
        with open('temp_result.json', 'w') as f:
            json.dump(result_json, f)
        coco_dt = coco_gt.loadRes('temp_result.json')
        cocoEval = COCOeval(coco_gt,coco_dt,'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval

@torch.no_grad()
def evaluate_nms_patch(model, data_loader, cfg, device, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in tqdm.tqdm(data_loader):
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)
        outputs = utils.post_processing(conf_thresh=0.001, nms_thresh=0.5, output=outputs)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, output in zip(images, targets, outputs):
            img_height, img_width = img.shape[:2]
            #human_box = target['human_box']
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = output[:,:4]
            scores = output[:,-2]
            labels = output[:,-1]

            human_box = target['human_box'].cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width + human_box[0]
            boxes[...,1] = boxes[...,1]*img_height + human_box[1]
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32).unsqueeze(0)
            # confs = output[...,4:].copy()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator

if __name__ == '__main__':
    cfg = Cfg
    cfg.gpu = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = 'checkpoints/Yolov4_modanet_128_epoch104.pth'
    json_gt = os.path.expanduser('~/data/datasets/modanet/Annots/modanet_instances_val_new.json')

    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
    else:
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    model.load_model(checkpoint, device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    #model.load_state_dict(torch.load(checkpoint), strict=True)

    val_dataset = YoloModanetHumanDataset(cfg.anno_path, cfg, train=False)

    n_val = len(val_dataset)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                            pin_memory=True, drop_last=False, collate_fn=val_collate)

    evaluate_nms(model, val_loader, cfg, device, human_patch=True, json_gt=None)
    #evaluate_nms(model, val_loader, cfg, device, human_patch=True, json_gt=json_gt)