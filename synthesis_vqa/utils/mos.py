import cv2
import json
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
# from videoenhancersdk.sdk_utils import torch_load_weights
import torchvision.transforms as transforms

try:
    from .utils import extract_frame_list_from_cfg
except:
    from utils.utils import extract_frame_list_from_cfg
try:
    from .mos_model import CenseoIVQAModel
except:
    from mos_model import CenseoIVQAModel


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return torch.rot90(x, 1, dims=[2,3])#TF.rotate(x, self.angle)

def init_mos_model(model_path=None, device='cpu'):
    model = CenseoIVQAModel(pretrained=False)
    
    if model_path.endswith('.enc'):
        model_weight = torch_load_weights(model_path)
    else:
        model_weight = torch.load(model_path)
        # model_weight = model_resume['model']
    model.load_state_dict(model_weight, strict=True)
    model = model.to(device)

    return model

class Mos(object):
    def __init__(self, Mos_model_path, device):
        # device_id = torch.cuda.current_device()
        self.device = device
        self.model = init_mos_model(Mos_model_path, device)
        # self.load_network(Iqa_model_path)
        self.model.eval()
        self.transform = None #transforms.Compose([
        #         transforms.Resize((224, 224)), 
        # ])


    def process(self, tensor_info, task_id):
        frame_ind = tensor_info.get('inpIds')
        image_info = {}
        frame = tensor_info.get("frame_tensor")
        h, w = frame.shape[2:]
        sub_img_dim = (720, 1280)
        resize_dim = (1080, 1920)
        resize_h, resize_w = resize_dim

        tran_list = []
        if (w - h) * (resize_w - resize_h) < 0:
            tran_list.append(MyRotationTransform(90))

        tran_list.append(transforms.Resize(resize_dim))
        tran_list.append(transforms.CenterCrop(sub_img_dim))

        self.transform = transforms.Compose(tran_list)

        inp = self.transform(frame)
        inp = inp[:,[2,1,0],:,:] #rgb2bgr

        with torch.no_grad():

            image_info['mos'] = self.model(inp).item()


        return image_info


