from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50,cfg_mobilenetv2
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface_mbv2 import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str,
                    help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
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


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    #if args.network == "mobile0.25":
    #    cfg = cfg_mnet
    #elif args.network == "resnet50":
    #    cfg = cfg_re50
    #elif args.network == "shuffle":
    cfg = cfg_mobilenetv2
    # use the 80,90,96 epoch model and calcuate for validation dataset and test dataset
    for index_s in range(1):
        if index_s==0:
            args.trained_model="./mbv2_widerface/mobilenetv2_epoch_150.pth"
            args.save_folder='./widerface_evaluate/twiderface_txt_150/'
            testset_folder = './data/widerface/val/images/'
            testset_list = testset_folder[:-7] + "wider_val.txt"
          

        net = RetinaFace(cfg=cfg, phase='test')
        net = load_model(net, args.trained_model, args.cpu)
        net.eval()
        print('index_s is Finished loading model!',index_s,testset_list,args.trained_model)
        #print(net)
        cudnn.benchmark = True
        # device = torch.device("cpu" if args.cpu else "cuda")

        device = torch.device("cuda")
        net = net.to(device)

        # testing dataset
        #testset_folder = args.dataset_folder
       # testset_list = args.dataset_folder[:-7] + "wider_val.txt"

        with open(testset_list, 'r') as fr:
            test_dataset = fr.read().split()
        num_images = len(test_dataset)

        _t = {'forward_pass': Timer(), 'misc': Timer()}

        # testing begin
        for i, img_name in enumerate(test_dataset):
            image_path = testset_folder + img_name
            print("***  imagepath", image_path)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)

            # testing scale
            scale_list = [500, 800, 1100, 1400, 1700]
            target_size = 800
            max_size = 1200
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            scales = [float(scale) / target_size * im_scale for scale in scale_list]
            list_all = []
            img_temp = img.copy()
            for resize_i in scales:
                resize = resize_i
                print("resiz e param", resize)

                img = cv2.resize(img_temp, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                #  cv2.imwrite("./ori.jpg",img)
                im_height, im_width, _ = img.shape
                print("image is", im_height, im_width)
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                _t['forward_pass'].tic()
                loc, conf, landms,head_cls_y,head_cls_p,head_cls_r = net(img)  # forward pass
                _t['forward_pass'].toc()
                _t['misc'].tic()
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                print("boxes original", boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3])
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
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

                # keep top-K before NMS
                order = scores.argsort()[::-1]
                # order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]
                # do NMS
                dets_single = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                # dets_all=np.vstack(dets_all,dets).astype(np.float32, copy=False)
                print(dets_single.shape)
                list_temp = dets_single.tolist()
                list_all += list_temp
                flip = True
                if (flip == True):
                    img1 = cv2.resize(img_temp, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                    img2 = img1.copy()
                    img2=cv2.flip(img2,1)
                    #   cv2.imwrite("./flip.jpg",img2)
                    im_height, im_width, _ = img2.shape
                    print("flip image is", im_height, im_width)
                    scale = torch.Tensor([img2.shape[1], img2.shape[0], img2.shape[1], img2.shape[0]])
                    img2 -= (104, 117, 123)
                    img2 = img2.transpose(2, 0, 1)
                    img2 = torch.from_numpy(img2).unsqueeze(0)
                    img2 = img2.to(device)
                    scale = scale.to(device)

                    _t['forward_pass'].tic()
                    loc, conf, landms,head_cls_y,head_cls_p,head_cls_r = net(img2)  # forward pass
                    _t['forward_pass'].toc()
                    _t['misc'].tic()
                    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                    priors = priorbox.forward()
                    priors = priors.to(device)
                    prior_data = priors.data
                    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                    # deal the flip data
                    boxes = boxes * scale
                    print("the flip x1,y1,x2,y2: ", boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3])

                    oldx1 = boxes[:, 0].clone()
                    oldx2 = boxes[:, 2].clone()
                    boxes[:, 0] = im_width - oldx2 - 1
                    boxes[:, 2] = im_width - oldx1 - 1
                    #  print("---------the flip x1,y1,x2,y2: ",boxes[0,0],boxes[0,1],boxes[0,2],boxes[0,3])

                    boxes = boxes / resize
                    print("---------the flip x1,y1,x2,y2: ", boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3])

                    boxes = boxes.cpu().numpy()
                    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                    scale1 = torch.Tensor([img2.shape[3], img2.shape[2], img2.shape[3], img2.shape[2],
                                           img2.shape[3], img2.shape[2], img2.shape[3], img2.shape[2],
                                           img2.shape[3], img2.shape[2]])
                    scale1 = scale1.to(device)
                    landms = landms * scale1 / resize
                    landms = landms.cpu().numpy()

                    # ignore low scores
                    inds = np.where(scores > args.confidence_threshold)[0]
                    boxes = boxes[inds]
                    landms = landms[inds]
                    scores = scores[inds]

                    # keep top-K before NMS
                    order = scores.argsort()[::-1]
                    # order = scores.argsort()[::-1][:args.top_k]
                    boxes = boxes[order]
                    landms = landms[order]
                    scores = scores[order]
                    # do NMS
                    dets_single = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    # dets_all=np.vstack(dets_all,dets).astype(np.float32, copy=False)
                    print(dets_single.shape)
                    list_temp = dets_single.tolist()
                    list_all += list_temp

            dets = np.array(list_all)
            print("dets shape", dets.shape)
            if (dets.shape[0] < 2):
                continue
            dets=bbox_vote(dets)

            _t['misc'].toc()

            # --------------------------------------------------------------------
            save_name = args.save_folder + img_name[:-4] + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)

            print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                         _t['forward_pass'].average_time,
                                                                                         _t['misc'].average_time))

