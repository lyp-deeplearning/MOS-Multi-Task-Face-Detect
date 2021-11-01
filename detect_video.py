from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
from config import cfg_mos_m,cfg_mos_s
#from layers.functions.prior_box import PriorBox
from utils.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.MOS import MOS
# import math
from math import cos, sin
from utils.box_utils import decode, decode_landm



parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./test_weights/MOS-M.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='cfg_mos_m', help='Backbone network cfg_mos_m or cfg_mos_s')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=840, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.55, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.55, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img




if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    net = None

    if args.network == "cfg_mos_m":
        cfg = cfg_mos_m
    elif args.network == "cfg_mos_s":
        cfg = cfg_mos_s

    #cfg = cfg_mos_m
    cfg = cfg
    net = MOS(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    #device = torch.device("cpu" if args.cpu else "cuda")
    device=torch.device("cuda")
    net = net.to(device)

    image_path = "./figures/4_Dancing_Dancing_4_85.jpg"
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    # testing scale
    target_size = args.long_side
    max_size = args.long_side
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args.origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    img_rgb = img_raw.copy()
    im_height, im_width, _ = img.shape
    print(im_height, im_width)

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms, head_cls_y, head_cls_p, head_cls_r = net(img)  # forward pass
    tic1 = time.time() - tic

    head_cls_y = head_cls_y.squeeze(0)
    head_cls_p = head_cls_p.squeeze(0)
    head_cls_r = head_cls_r.squeeze(0)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    head_cls_y = torch.sum(head_cls_y * idx_tensor, 1).to(device) * 3 - 99
    head_cls_p = torch.sum(head_cls_p * idx_tensor, 1).to(device) * 3 - 99
    head_cls_r = torch.sum(head_cls_r * idx_tensor, 1).to(device) * 3 - 99

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])

    head_cls_y = head_cls_y.cpu().numpy()
    head_cls_p = head_cls_p.cpu().numpy()
    head_cls_r = head_cls_r.cpu().numpy()

    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    head_cls_y = head_cls_y[inds]
    head_cls_p = head_cls_p[inds]
    head_cls_r = head_cls_r[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    head_cls_y = head_cls_y[order]
    head_cls_p = head_cls_p[order]
    head_cls_r = head_cls_r[order]
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.array(idx_tensor)

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    yaw_predicted = head_cls_y[keep]
    pitch_predicted = head_cls_p[keep]
    roll_predicted = head_cls_r[keep]

    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]
    yaw_predicted = yaw_predicted[:args.keep_top_k]
    pitch_predicted = pitch_predicted[:args.keep_top_k]
    roll_predicted = roll_predicted[:args.keep_top_k]

    dets = np.concatenate((dets, landms), axis=1)
    for i in range(len(dets)):
        b = dets[i]
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_rgb, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        cx = b[0]
        cy = b[1] + 12


        text = "yaw:" + str(int(yaw_predicted[i]))  # + "," + "p:" + str(int(pitch_predicted[i])) + "," + "r:" + str(
        # int(roll_predicted[i]))

        cv2.putText(img_rgb, text, (cx - 10, cy - 25),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 0, 255))
        # landms
        cv2.circle(img_rgb, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_rgb, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_rgb, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_rgb, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_rgb, (b[13], b[14]), 1, (255, 0, 0), 4)
        draw_axis(img_rgb, int(yaw_predicted[i]), int(pitch_predicted[i]), int(roll_predicted[i]), tdx=b[9],
                  tdy=b[10], size=30)


    #cv2.imshow("frame", img_rgb)
    cv2.imwrite("./figures/result.jpg", img_rgb)








