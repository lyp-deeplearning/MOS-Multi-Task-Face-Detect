import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from models.mobilenetv2 import BasicMobileNet
import torch.utils.model_zoo as model_zoo

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
from models.net import conv_bn as conv_bn
from models.cross import *

class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)
class ClassHead_stitch(nn.Module):
    def __init__(self, inchannels=512):
        super(ClassHead_stitch, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, inchannels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        return out

class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)
class BboxHead_stitch(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead_stitch, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, inchannels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)

        return out

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)

class LandmarkHead_stitch(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead_stitch, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, inchannels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        return out
## 1  pose head to predict the pose angle
class HeadPose(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(HeadPose, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels * 2, num_anchors * 66, kernel_size=(1, 1), stride=1, padding=0)
        self.Conv_pose = conv_bn(inchannels, inchannels * 2, stride=1)

    def forward(self, x):
        out = self.Conv_pose(x)
        out = self.conv1x1(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 66)

class HeadPose_stitch(nn.Module):
    def __init__(self, inchannels=512):
        super(HeadPose_stitch, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels , inchannels, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        return out

"""
1 load the official model but the layer needs to name again
"""
def load_model(model, state_dict):
    new_model = model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)


class MOS(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(MOS, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mos_s':
            import torchvision.models as models
            backbone = models.shufflenet_v2_x1_0(pretrained=True)
        elif cfg['name'] == 'mos_m':
            import torchvision.models as models
            #### 1 load model from official pytorch model zoo
            backbone=models.mobilenet_v2(pretrained=True)
            #### 2 load model from the own file
            backbone=BasicMobileNet()
            if cfg['pretrain'] == True:
                pretrain_model = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',progress=True)
                load_model(backbone,pretrain_model)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        #### mbv2 stage 4-6 return channels number
        if cfg['name'] == 'mos_s':
            in_channels_list = [116,232,464]
        elif cfg['name'] == 'mos_m':
            in_channels_list = [32, 96, 320]
        print(cfg['name'])
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        # 2  add pose head conv module
        self.Conv_pose = conv_bn(out_channels, out_channels * 2, stride=1)

        self.stitch_classhead=self._make_stitch_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.stitch_BboxHead = self._make_stitch_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.stitch_LandmarkHead = self._make_stitch_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.stitch_PoseHead_y = self._make_stitch_pose_head_y(fpn_num=3, inchannels=cfg['out_channel'])
        self.stitch_PoseHead_p = self._make_stitch_pose_head_p(fpn_num=3, inchannels=cfg['out_channel'])
        self.stitch_PoseHead_r = self._make_stitch_pose_head_r(fpn_num=3, inchannels=cfg['out_channel'])


        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.PoseHead_y = self._make_pose_head_y(fpn_num=3, inchannels=cfg['out_channel'])
        self.PoseHead_p = self._make_pose_head_p(fpn_num=3, inchannels=cfg['out_channel'])
        self.PoseHead_r = self._make_pose_head_r(fpn_num=3, inchannels=cfg['out_channel'])
        # stitch param
        self.stages= ['fpn1', 'fpn2', 'fpn3']
        self.channels=cfg['out_channel']
        self.tasks=['classhead','bboxhead','landmarkhead','pose_y','pose_p','pose_r']
        # cross stitch
        alpha = 0.9
        beta = 0.1
        channels = self.channels
        self.cross_stitch = nn.ModuleDict(
            {stage: CrossStitchUnit(self.tasks, channels, alpha, beta) for stage in self.stages})


    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    ## stitch class head *3
    def _make_stitch_class_head(self,fpn_num=3,inchannels=64):
        stitch_classhead = nn.ModuleList()
        for i in range(fpn_num):
            stitch_classhead.append(ClassHead_stitch(inchannels))
        return stitch_classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead
    ## stitch box head *3
    def _make_stitch_bbox_head(self, fpn_num=3, inchannels=64):
        stitch_bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            stitch_bboxhead.append(BboxHead_stitch(inchannels))
        return stitch_bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    ## stitch landmark head *3
    def _make_stitch_landmark_head(self, fpn_num=3, inchannels=64):
        stitch_landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            stitch_landmarkhead.append(LandmarkHead_stitch(inchannels))
        return stitch_landmarkhead

    def _make_pose_head_y(self, fpn_num=3, inchannels=64, anchor_num=2):
        posehead_y = nn.ModuleList()
        for i in range(fpn_num):
            posehead_y.append(HeadPose(inchannels, anchor_num))
        return posehead_y

    ## stitch posey head *3
    def _make_stitch_pose_head_y(self, fpn_num=3, inchannels=64):
        stitch_posehead_y = nn.ModuleList()
        for i in range(fpn_num):
            stitch_posehead_y.append(HeadPose_stitch(inchannels))
        return stitch_posehead_y

    def _make_pose_head_p(self, fpn_num=3, inchannels=64, anchor_num=2):
        posehead_p = nn.ModuleList()
        for i in range(fpn_num):
            posehead_p.append(HeadPose(inchannels, anchor_num))
        return posehead_p

    ## stitch posep head *3
    def _make_stitch_pose_head_p(self, fpn_num=3, inchannels=64):
        stitch_posehead_p = nn.ModuleList()
        for i in range(fpn_num):
            stitch_posehead_p.append(HeadPose_stitch(inchannels))
        return stitch_posehead_p

    def _make_pose_head_r(self, fpn_num=3, inchannels=64, anchor_num=2):
        posehead_r = nn.ModuleList()
        for i in range(fpn_num):
            posehead_r.append(HeadPose(inchannels, anchor_num))
        return posehead_r
    ## stitch poser head *3
    def _make_stitch_pose_head_r(self, fpn_num=3, inchannels=64):
        stitch_posehead_r = nn.ModuleList()
        for i in range(fpn_num):
            stitch_posehead_r.append(HeadPose_stitch(inchannels))
        return stitch_posehead_r

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]
        #stitch
        sbbox_regressions=[]
        sclassifications=[]
        sldm_regressions=[]
        shead_cls_y=[]
        shead_cls_p=[]
        shead_cls_r=[]
        for i in range(3):
            sbbox_regressions.append(self.stitch_BboxHead[i](features[i]))
            sclassifications.append(self.stitch_classhead[i](features[i]))
            sldm_regressions.append(self.stitch_LandmarkHead[i](features[i]))

            # pose head
            shead_cls_y.append(self.stitch_PoseHead_y[i](features[i]))
            shead_cls_p.append(self.stitch_PoseHead_p[i](features[i]))
            shead_cls_r.append(self.stitch_PoseHead_r[i](features[i]))
        s1_temp=[sbbox_regressions[0],sclassifications[0],sldm_regressions[0],shead_cls_y[0],shead_cls_p[0],shead_cls_r[0]]
        s2_temp=[sbbox_regressions[1],sclassifications[1],sldm_regressions[1],shead_cls_y[1],shead_cls_p[1],shead_cls_r[1]]
        s3_temp=[sbbox_regressions[2],sclassifications[2],sldm_regressions[2],shead_cls_y[2],shead_cls_p[2],shead_cls_r[2]]


        #x1 = {task: x for task in self.tasks}  # Feed as input to every single-task network
        x1 = {task: s1_temp[i] for i, task in enumerate(self.tasks)}  # Feed as input to every single-task network
        x2={task: s2_temp[i] for i, task in enumerate(self.tasks)}  # Feed as input to every single-task network
        x3={task: s3_temp[i] for i, task in enumerate(self.tasks)}  # Feed as input to every single-task network
        x=[x1,x2,x3]
        s_o=[]

        # Backbone
        for i,stage in enumerate(self.stages):
            # Forward through next stage of task-specific network
            s_o.append(self.cross_stitch[stage](x[i]))
        bbox_features=[s_o[0]['bboxhead'],s_o[1]['bboxhead'],s_o[2]['bboxhead']]
        classifications_features=[s_o[0]['classhead'],s_o[1]['classhead'],s_o[2]['classhead']]
        ldm_regressions_features=[s_o[0]['landmarkhead'],s_o[1]['landmarkhead'],s_o[2]['landmarkhead']]
        head_cls_y_features=[s_o[0]['pose_y'],s_o[1]['pose_y'],s_o[2]['pose_y']]
        head_cls_p_features=[s_o[0]['pose_p'],s_o[1]['pose_p'],s_o[2]['pose_p']]
        head_cls_r_features=[s_o[0]['pose_r'],s_o[1]['pose_r'],s_o[2]['pose_r']]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(bbox_features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(classifications_features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(ldm_regressions_features)], dim=1)

        # pose head
        head_cls_y = torch.cat([self.PoseHead_y[i](feature) for i, feature in enumerate(head_cls_y_features)], dim=1)
        head_cls_p = torch.cat([self.PoseHead_p[i](feature) for i, feature in enumerate(head_cls_p_features)], dim=1)
        head_cls_r = torch.cat([self.PoseHead_r[i](feature) for i, feature in enumerate(head_cls_r_features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions, head_cls_y, head_cls_p, head_cls_r)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions,
                      F.softmax(head_cls_y, dim=-1), F.softmax(head_cls_p, dim=-1), F.softmax(head_cls_r, dim=-1))
        return output
