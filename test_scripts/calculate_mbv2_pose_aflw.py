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
from data import cfg_mnet, cfg_re50,cfg_mobilenetv2
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface_mbv2 import RetinaFace
import math
from math import cos, sin
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import torch.nn.functional as F
import matplotlib.cm
import copy

# from utils_render.renderer import Renderer
# from utils_render.image_operations import expand_bbox_rectangle
# from utils_render.pose_operations import get_pose
# from scipy.spatial.transform import Rotation
# from matplotlib import pyplot as plt
import utils1

parser = argparse.ArgumentParser(description='Test')
####/home/lyp/paper_experiments/pre_data/train_300wlp/加入全部数据/1/Resnet50_epoch_22.pth  300wlp
####/home/lyp/paper_experiments/pre_data/train_300wlp/train_aflw/Resnet50_epoch_46.pth
parser.add_argument('-m', '--trained_model',
                    default='/home/face-detect-trainmodel/test_pose_weights/mobilenetv2_epoch_149.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='shuffle_0.5', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=840, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.4, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.2, type=float, help='visualization_threshold')
args = parser.parse_args()
def get_pose(vertices, twod_landmarks, camera_intrinsics, initial_pose=None):
    threed_landmarks = vertices
    twod_landmarks = np.asarray(twod_landmarks).astype("float32")

    # if initial_pose is provided, use it as a guess to solve new pose
    if initial_pose is not None:
        initial_pose = np.asarray(initial_pose)
        retval, rvecs, tvecs = cv2.solvePnP(
            threed_landmarks,
            twod_landmarks,
            camera_intrinsics,
            None,
            rvec=initial_pose[:3],
            tvec=initial_pose[3:],
            flags=cv2.SOLVEPNP_EPNP,
            useExtrinsicGuess=True,
        )
    else:
        retval, rvecs, tvecs = cv2.solvePnP(
            threed_landmarks,
            twod_landmarks,
            camera_intrinsics,
            None,
            flags=cv2.SOLVEPNP_EPNP,
        )

    rotation_mat = np.zeros(shape=(3, 3))
    R = cv2.Rodrigues(rvecs, rotation_mat)[0]

    RT = np.column_stack((R, tvecs))
    P = np.matmul(camera_intrinsics, RT)
    dof = np.append(rvecs, tvecs)

    return P, dof
def bbox_is_dict(bbox):
    # check if the bbox is a not dict and convert it if needed
    if not isinstance(bbox, dict):
        temp_bbox = {}
        temp_bbox["left"] = bbox[0]
        temp_bbox["top"] = bbox[1]
        temp_bbox["right"] = bbox[2]
        temp_bbox["bottom"] = bbox[3]
        bbox = temp_bbox

    return bbox


def get_bbox_intrinsics(image_intrinsics, bbox):
    # crop principle point of view
    bbox_center_x = bbox["left"] + ((bbox["right"] - bbox["left"]) // 2)
    bbox_center_y = bbox["top"] + ((bbox["bottom"] - bbox["top"]) // 2)

    # create a camera intrinsics from the bbox center
    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics


def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    # check if bbox is np or dict
    bbox = bbox_is_dict(bbox)

    # rotation vector
    rvec = pose[:3].copy()

    # translation and scale vector
    tvec = pose[3:].copy()

    # get camera intrinsics using bbox
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    # focal length
    focal_length = image_intrinsics[0, 0]

    # bbox_size
    bbox_width = bbox["right"] - bbox["left"]
    bbox_height = bbox["bottom"] - bbox["top"]
    bbox_size = bbox_width + bbox_height

    # adjust scale
    tvec[2] *= focal_length / bbox_size

    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(tvec.T)

    # reverse the projected points using the full image camera intrinsics
    tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))

    # same for rotation
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(rmat)
    # reverse the projected points using the full image camera intrinsics
    rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    return np.concatenate([rvec, tvec])

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


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

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
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

p_in = []
p_out = []

#
def hook_fn(module, inputs, outputs):
    p_in.append(inputs)
    p_out.append(outputs)

def put_heatmap_on_image(ori_image, activation, colormap_name):
    """
    ori_image (PIL image): 原始图像
    activation (numpy arr): 即上面得到的p2_logits
    colormap_name (str): 采用何种matplotlib.cm的colormap
    """
    # colormap
    color_map = matplotlib.cm.get_cmap(colormap_name)
    # 把colormap添加到activation，即activation的以何种
    # colormap进行显示
    no_trans_heatmap = color_map(activation)
    # 添加alpha通道，即透明度
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    #
    heatmap_on_image = Image.new("RGBA", ori_image.size)
    heatmap_on_image = Image.alpha_composite(
    					heatmap_on_image, ori_image.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(
    					heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def rot2Euler(imgpath, rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1) - 0.8356857

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2) + 0.005409

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4) - 2.573345436

    # 单位转换：将弧度转换为度
    pitch_degree = int((pitch / math.pi) * 180)
    yaw_degree = int((yaw / math.pi) * 180)
    roll_degree = int((roll / math.pi) * 180)

    #drawResult(imgpath, yaw, pitch, roll, save_dir)

    print("Radians:")
    print("Yaw:", yaw_degree)
    print("Pitch:", pitch_degree)
    print("Roll:", roll_degree)
    str_angle=[yaw_degree,pitch_degree,roll_degree]
    return  str_angle


def headPosEstimate(imgpath, landmarks):

    # 3D model points
    model_3d_points = np.array(([-165.0, 170.0, -115.0],  # Left eye
								[165.0, 170.0, -115.0],   # Right eye
								[0.0, 0.0, 0.0],          # Nose tip
								[-150.0, -150.0, -125.0], # Left Mouth corner
								[150.0, -150.0, -125.0]), dtype=np.double) # Right Mouth corner)
    landmarks.dtype = np.double
    # Camera internals
    img = cv2.imread(imgpath)
    img_size = img.shape
    focal_length = img_size[1]
    center =  [img_size[1]/2, img_size[0]/2]
    camera_matrix = np.array(([focal_length, 0, center[0]],
							[0, focal_length, center[1]],
							[0, 0, 1]),dtype=np.double)


    dist_coeffs = np.array([0,0,0,0], dtype=np.double)
    found, rotation_vector, translation_vector = cv2.solvePnP(model_3d_points, landmarks, camera_matrix, dist_coeffs)

    angle_result=rot2Euler(imgpath,rotation_vector)
    return angle_result





if __name__ == '__main__':
    ## 1 reneder init


    torch.set_grad_enabled(False)

    cfg = None
    net = None

    cfg = cfg_mobilenetv2
    net = RetinaFace(cfg=cfg, phase='test')

    ##################################################
    # 1 read the pose test model
    model_path = "/home/face-detect-trainmodel/test_pose_weights/"
    picture_path = os.listdir("/home/face-detect-trainmodel/POSE_AFLW2000/picture_aflw1980/")
    model_testings = os.listdir(model_path)
    f_in = open("./pose_result.txt", "a+")
    for model_name in model_testings:
        ## load the model
        pose_model_path="/home/face-detect-trainmodel/test_pose_weights/"+model_name
        net = load_model(net, pose_model_path, args.cpu)
        net.eval()
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        # device = torch.device("cpu" if args.cpu else "cuda")
        device = torch.device("cuda")
        net = net.to(device)
        flag = True
        padding=False
        ## read the image
        image_list=[300,350,400,450,500,550,600,640,800]
        for index in range(len(image_list)):
            resize_width=image_list[index]
            if image_list[index]==800:
                padding=True
            # initial parameter
            all_pitch_error = 0
            all_yaw_error = 0
            all_roll_error = 0
            real_p = 0
            real_y = 0
            real_r = 0
            real_number = 0
            zero_detect=0
            for index in range(len(picture_path)):
                picture_name = picture_path[index]
                image_path = "/home/face-detect-trainmodel/POSE_AFLW2000/picture_aflw1980/" + picture_name

                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                matname = picture_name.split('.')[0] + ".mat"
                # show image
                mat_path = os.path.join("/home/face-detect-trainmodel/POSE_AFLW2000/pose_label/", matname)

                img = np.float32(img_raw)
                # testing scale
                target_size = args.long_side
                max_size = args.long_side
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])


                # prevent bigger axis from being more than max_size:
                # if np.round(resize * im_size_max) > max_size:
                #     resize = float(max_size) / float(im_size_max)
                # if args.origin_size:
                #     resize = 1
                #
                # if resize != 1:
                #     img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                if padding==False:
                    img = cv2.resize(img, (resize_width,resize_width))
                    resize = float(resize_width) / float(im_size_min)
                else:
                    mask1 = np.zeros((640, 640,3), dtype=np.float32) ## get a black mask
                    mask1[95:95+450,95:95+450]=img
                    img=mask1
                    resize = 1
                #cv2.imwrite("./the_black_image.jpg",img)
                img_rgb=img.copy()
                im_height, im_width, _ = img.shape
                #print(im_height,im_width)

                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)


                loc, conf, landms, head_cls_y, head_cls_p, head_cls_r = net(img)  # forward pass




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
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
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


                ## show the pose
                # We get the pose in radians
                pose = utils1.get_ypr_from_mat(mat_path)
                # And convert to degrees.
                pitch = pose[0] * 180 / np.pi
                yaw = pose[1] * 180 / np.pi
                roll = pose[2] * 180 / np.pi
                cont_labels = np.array([yaw, pitch, roll])

                if np.max(cont_labels) > 99.0 or np.min(cont_labels) < -99.0:
                    zero_detect += 1
                    continue
                real_i=0
                sss=0
                ss_id=0
                if (len(dets)==0):
                    zero_detect+=1
                    print("img is not detect",image_path)
                    continue

                # if len(dets)!=1:
                #     continue

                for ii in range(len(dets)):
                    b = dets[ii]
                    w=b[2]-b[0]
                    h=b[3]-b[1]
                    s_temp=w*h
                    if (s_temp>sss):
                        sss=s_temp
                        ss_id=ii




                real_i=ss_id
                # if len(dets)!=1:
                #     continue


                yaw_p=yaw_predicted[real_i]
                pitch_p=pitch_predicted[real_i]
                roll_p=roll_predicted[real_i]

                pitch_error = 0
                yaw_error = 0
                roll_error = 0

                pitch_error = np.abs(pitch - pitch_p)
                all_pitch_error += pitch_error
                real_p += 1

                yaw_error = np.abs(yaw - yaw_p)
                all_yaw_error += yaw_error

                roll_error = np.abs(roll -roll_p)
                all_roll_error += roll_error


            print(model_name,resize_width,index, real_p, all_pitch_error/real_p, all_yaw_error/real_p, all_roll_error/real_p,(all_pitch_error/real_p+all_yaw_error/real_p+all_roll_error/real_p)/3)
            print("zero detect",zero_detect)

            f_line=model_name+" "+str(resize_width)+"  p: ",str(all_pitch_error/real_p)+"  y: ",str(all_yaw_error/real_p)+"  r: ",str(all_roll_error/real_p)+"  mae: ",str((all_pitch_error/real_p+all_yaw_error/real_p+all_roll_error/real_p)/3)+"\n"
            f_in.writelines(f_line)













