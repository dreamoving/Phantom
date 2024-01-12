import torch, os, cv2
import numpy as np

try:
    from .utils.mos import Mos
except:
    from utils.mos import Mos

class VideoQualitySynthesisAnalysis(object):
    def __init__(self, mos_weights, device=torch.device(0), vqa_cfg_file='./vqa_cfg.json'):
        
        self.devide = device
        self.vqa_cfg_file = vqa_cfg_file
        
        self.mos_model = Mos(mos_weights, device)


    def extract_frame_list(self, video_info, *model_list):
        for model in model_list:
            model.extract_frame_list(self.vqa_cfg_file, video_info)
    
    def get_model_list(self, vqa_mode, video_info):
        task_id = video_info.get('task_id')
        vqa_model = vqa_mode.split(',')

        video_info['vqa_dict'] = ['vqa_'+model+'_info' for model in vqa_model]
        return vqa_model



    def process_image(self, tensor_info, image_info, vqa_mode='normal', ):
        task_id = image_info.get('task_id')
        model_name_list = self.get_model_list(vqa_mode, image_info)
        
        model_list = []
        for model_name in model_name_list:
            model_list.append(eval('self.'+model_name+'_model'))

        frame = tensor_info.get("frame")

        for model_name, model in zip(model_name_list, model_list):
            tensor_info["frame_tensor"] = torch.from_numpy(frame.astype(np.float64)/255).to('cuda').permute((2,0,1)).unsqueeze(0).float()
            model_image_info = model.process(tensor_info, task_id)
            image_info['vqa_'+model_name+'_info'] = model_image_info

        return image_info

    def tensor_iterator(self, input_image, device):
        frame = cv2.imread(input_image)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return  {"frame":frame, "inpIds":0}
                
        
        # print('iter test ####')
    def run(self, input_image, image_info, vqa_mode):
        if isinstance(input_image, str):
            tensor_info = self.tensor_iterator(input_image, device='cuda')
        else:
            tensor_info = {"frame": input_image, "inpIds":0}
        image_info = self.process_image(tensor_info, image_info, vqa_mode)
        
        return image_info
