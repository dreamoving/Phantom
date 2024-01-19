# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from cnstd import CnStd
import warnings
warnings.filterwarnings('ignore')


from .models import networks, ocr
import torchvision.transforms as transforms
from .utils.alphabets import alphabet
from .utils import regionmerge as rm
import traceback

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

cur_path = os.path.dirname(os.path.abspath(__file__))

def print_networks(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6

def get_labels_from_text(text):
    labels = []
    for t in text:
        index = alphabet.find(t)
        labels.append(index)
    return labels

def get_text_from_labels(TestPreds):
    PredsText = ''
    for i in range(len(TestPreds)):
        PredsText = PredsText + alphabet[TestPreds[i]]
    return PredsText

def clear_labels(TestPreds):
    labels = []
    PredsInds = torch.max(TestPreds.detach(), 1)[1]
    for i in range(PredsInds.size(0)):
        if (not (i > 0 and PredsInds[i - 1] == PredsInds[i])) and PredsInds[i] < len(alphabet):
            labels.append(PredsInds[i])
    return labels


class TextRestoration(object):
    def __init__(self, device='cuda', use_new_bbox=True, use_real_ocr=False):
        self.device = device
        self.use_new_bbox = use_new_bbox
        self.use_real_ocr = use_real_ocr

        self.modelTSPGAN = networks.TSPGAN()
        self.modelTSPGAN.load_state_dict(torch.load(os.path.join(cur_path, 'checkpoints/net_prior_generation.pth'))['params'], strict=True)
        self.modelTSPGAN.eval()

        self.modelSR = networks.TSPSRNet()
        self.modelSR.load_state_dict(torch.load(os.path.join(cur_path, 'checkpoints/net_sr.pth'))['params'], strict=True)
        self.modelSR.eval()

        self.modelEncoder = networks.TextContextEncoderV2()
        self.modelEncoder.load_state_dict(torch.load(os.path.join(cur_path, 'checkpoints/net_transformer_encoder.pth'))['params'], strict=True)
        self.modelEncoder.eval()

        if use_new_bbox:
            self.modelBBox = ocr.TransformerOCR(use_new_bbox=True)
            self.modelBBox.load_state_dict(torch.load(os.path.join(cur_path, 'checkpoints/net_new_bbox.pth'))['params'], strict=True)
            self.modelBBox.eval()
            self.modelBBox = self.modelBBox.to(device)

        if use_real_ocr:
            self.modelOCR = ocr.TransformerOCR()
            self.modelOCR.load_state_dict(torch.load(os.path.join(cur_path, 'checkpoints/net_real_world_ocr.pth'))['params'], strict=True)
            self.modelOCR.eval()
            self.modelOCR = self.modelOCR.to(device)
            
        self.std = CnStd(model_name='db_resnet34',rotated_bbox=True, model_backend='pytorch', box_score_thresh=0.3, min_box_size=10, context=device, root=os.path.join(cur_path, 'checkpoints/db_resnet34'))
        self.insize = 32

        self.modelTSPGAN = self.modelTSPGAN.to(device)
        self.modelSR = self.modelSR.to(device)
        self.modelEncoder = self.modelEncoder.to(device)


        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

        torch.cuda.empty_cache()


    def handle_texts(self, img, bg=None, sf=4, is_aligned=False, region_merge='meanstd'): # img should be np.array RGB 0-255
        height, width = img.shape[:2]
        bg_height, bg_width = bg.shape[:2]
        box_infos = self.std.detect(img)

        full_mask = np.zeros(bg.shape, dtype=np.float32)
        full_img = np.zeros(bg.shape, dtype=np.float32) #+255

        orig_texts, enhanced_texts = [], []
        img_merged = bg.copy()

        if not is_aligned:
            for i, box_info in enumerate(box_infos['detected_texts']):
                box = box_info['box'].astype(np.int32)# left top, right top, right bottom, left bottom, [width, height]
                std_cropped = box_info['cropped_img']

                h, w = std_cropped.shape[:2]
                score = box_info['score']
                print(score)
                if score < 0.3 or w < 8 or h < 8 or h > 35:
                    continue
                
                scale_wl = 0.04#0.04
                scale_hl = 0.04
                
                move_w = (box[0][0] + box[2][0]) * (scale_wl) / 2
                move_h = (box[0][1] + box[2][1]) * (scale_hl) / 2

                extend_box = box.copy()

                extend_box[0][0] = extend_box[0][0] * (1+scale_wl) - move_w
                extend_box[0][1] = extend_box[0][1] * (1+scale_hl) - move_h
                extend_box[1][0] = extend_box[1][0] * (1+scale_wl) - move_w
                extend_box[1][1] = extend_box[1][1] * (1+scale_hl) - move_h
                extend_box[2][0] = extend_box[2][0] * (1+scale_wl) - move_w
                extend_box[2][1] = extend_box[2][1] * (1+scale_hl) - move_h
                extend_box[3][0] = extend_box[3][0] * (1+scale_wl) - move_w
                extend_box[3][1] = extend_box[3][1] * (1+scale_hl) - move_h

                if w > h:
                    ref_h = self.insize
                    ref_w = int(ref_h * w / h)
                else:
                    ref_w = self.insize
                    ref_h = int(ref_w * h / w)
                ref_point = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]])
                det_point = np.float32(extend_box)

                matrix = cv2.getPerspectiveTransform(det_point, ref_point)
                inv_matrix = cv2.getPerspectiveTransform(ref_point*sf, det_point*sf)

                cropped_img = cv2.warpPerspective(img, matrix, (ref_w, ref_h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
                in_img = cropped_img

                SQ = self.text_sr(in_img)
                if SQ is None:
                    continue

                if sf != 4:
                    SQ = cv2.resize(SQ, None, fx=sf/4.0, fy=sf/4.0, interpolation=cv2.INTER_AREA)

                # LQ = transforms.ToTensor()(in_img)
                # LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
                # LQ = LQ.unsqueeze(0)
                # SQ = self.modelText(LQ.to(self.device))
                # SQ = SQ * 0.5 + 0.5
                # SQ = SQ.squeeze(0).permute(1, 2, 0) #.flip(2) # RGB->BGR
                # SQ = np.clip(SQ.float().cpu().numpy(), 0, 1) * 255.0
                orig_texts.append(in_img)
                enhanced_texts.append(SQ)

                if region_merge == 'blur':
                    tmp_mask = np.ones(SQ.shape).astype(np.float)*255

                    warp_mask = cv2.warpPerspective(tmp_mask, inv_matrix, (bg_width, bg_height), flags=3)
                    warp_img = cv2.warpPerspective(SQ, inv_matrix, (bg_width, bg_height), flags=3)
                    
                    full_img = full_img + warp_img
                    full_mask = full_mask + warp_mask
                else:
                    warp_img = cv2.warpPerspective(SQ, inv_matrix, (bg_width, bg_height), flags=3).astype(np.uint8)
                    
                    scaled_det_point = np.array(det_point) * sf
                    scaled_det_point = np.round(scaled_det_point).astype(int)

                    x_low = max(max(scaled_det_point[0][0], scaled_det_point[3][0]), 0)
                    x_high = min(min(scaled_det_point[1][0], scaled_det_point[2][0]), bg_width)
                    y_low = max(max(scaled_det_point[0][1], scaled_det_point[1][1]), 0)
                    y_high = min(min(scaled_det_point[2][1], scaled_det_point[3][1]), bg_height)

                    if x_low >= x_high or y_low >= y_high:
                        continue

                    img1_crop = warp_img[y_low:y_high, x_low:x_high]
                    img2_crop = bg[y_low:y_high, x_low:x_high]

                    img_transferred = rm.transfer_color(img1_crop, img2_crop, "meanstd")
                    img_merged = rm.merge(img_transferred, img_merged, [x_low, y_low, x_high-x_low, y_high-y_low], 'smooth_cleanedge')

            if region_merge == 'blur':
                index = full_mask>0
                full_img[index] = full_img[index]/full_mask[index]

                full_mask = np.clip(full_mask, 0, 1)
                kernel = np.ones((7, 7), dtype=np.uint8)
                full_mask_dilate = cv2.erode(full_mask, kernel, 1)

                full_mask_blur = cv2.GaussianBlur(full_mask_dilate, (3, 3), 0) 
                full_mask_blur = cv2.GaussianBlur(full_mask_blur, (3, 3), 0) 

                img = cv2.convertScaleAbs(bg*(1-full_mask_blur) + full_img*255*full_mask_blur)
            else:
                img = img_merged
            
        return img, orig_texts, enhanced_texts

    def text_sr(self, img):
        '''
        Step 1: Reading Image
        '''
        h, w, _ = img.shape
        # ShowLQ = cv2.resize(img, (0,0), fx=128/h, fy=128/h,  interpolation=cv2.INTER_CUBIC)
        LQ_org = cv2.resize(img, (0,0), fx=32/h, fy=32/h,  interpolation=cv2.INTER_CUBIC)
        segments_list = self.crop_image_to_segments(LQ_org)
        SR_list = []

        for LQ in segments_list:
            ori_lq_w = LQ.shape[1]
            img_seg = LQ.copy()

            TextLQFillBG = np.zeros((32, 32*16, 3)).astype(LQ.dtype)
            TextLQFillBG[:, :LQ.shape[-2], :] = TextLQFillBG[:, :LQ.shape[-2], :] + LQ
            LQ = TextLQFillBG

            LQ = transforms.ToTensor()(LQ)
            LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)

            LQ = LQ.unsqueeze(0)
            LQ = LQ.to(self.device)

            '''
            Step 2: Predicting the character labels, bounding boxes and font style.
            '''
            with torch.no_grad():
                preds_cls, preds_locs_l_r, w = self.modelEncoder(LQ)
            
            labels = clear_labels(preds_cls[0])
            pre_text = get_text_from_labels(labels)

            preds_locs = preds_locs_l_r.clone()
            for n in range(0, 16*2, 2):
                preds_locs[0][n] = (preds_locs_l_r[0][n+1] + preds_locs_l_r[0][n]) / 2.0 #center
                preds_locs[0][n+1] = (preds_locs_l_r[0][n+1] - preds_locs_l_r[0][n]) / 2.0 # width
            
            assert w.size(0) == 1
            w0 = w[:1,...].clone() #

            # use new ocr model
            result = self.ocr_recognition(img_seg)
            pre_text = result['text'][0]
            labels = get_labels_from_text(pre_text)

            # '''
            # Step 2.5: Predicting the character labels using real-world OCR model trained on real-world chinese dataset, see:
            # https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main
            # '''
            
            # if self.use_real_ocr:
            #     LQForOCR = cv2.resize(img_seg, (256, 32), interpolation=cv2.INTER_CUBIC)
            #     LQForOCR = transforms.ToTensor()(LQForOCR)
            #     LQForOCR = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQForOCR)

            #     LQForOCR = LQForOCR.unsqueeze(0)
            #     LQForOCR = LQForOCR.to(self.device)

            #     #---------------character classification--------------------
            #     max_length = 20
            #     batch = 1
            #     pred = torch.zeros(batch,1).long().cuda()
            #     image_features = None
            #     prob = torch.zeros(batch, max_length).float()
            #     for i in range(max_length):
            #         length_tmp = torch.zeros(batch).long().cuda() + i + 1
            #         with torch.no_grad():
            #             result = self.modelOCR(image=LQForOCR, text_length=length_tmp, text_input=pred, conv_feature=image_features, test=True)
            #         prediction = result['pred']
            #         now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
            #         prob[:,i] = torch.max(torch.softmax(prediction,2), 2)[0][:,-1]
            #         pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
            #         image_features = result['conv']

            #     text_pred_list = []
            #     now_pred = []
            #     for j in range(max_length):
            #         if pred[0][j] != 6737:
            #             now_pred.append(pred[0][j])
            #         else:
            #             break
            #     text_pred_list = torch.Tensor(now_pred)[1:].long().cuda()
            #     pre_text = ""
            #     for i in text_pred_list:
            #         if i == (len(alphabet)+2):
            #             continue
            #         pre_text += alphabet[i-2]

            #     labels = get_labels_from_text(pre_text)


            '''
            Step 2.75: Predicting the bbox using our synthtic images
            '''
            if self.use_new_bbox:
                LQForBBox = cv2.resize(img_seg, (256,32), interpolation=cv2.INTER_CUBIC)
                LQForBBox = transforms.ToTensor()(LQForBBox)
                LQForBBox = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQForBBox)

                LQForBBox = LQForBBox.unsqueeze(0)
                LQForBBox = LQForBBox.to(self.device)

                #---------------character classification--------------------
                max_length = 20
                batch = 1
                pred = torch.zeros(batch,1).long().cuda()
                loc = torch.zeros(batch,1).float().cuda()
                image_features = None
                for i in range(max_length):
                    length_tmp = torch.zeros(batch).long().cuda() + i + 1
                    with torch.no_grad():
                        result = self.modelBBox(image=LQForBBox, text_length=length_tmp, text_input=pred, conv_feature=image_features, test=True)
                    prediction = result['pred']
                    now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
                    pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
                    now_loc = result['loc'][:,-1].view(-1,1) #* self.opt['datasets']['train']['ocr_width'] # using sigmoid, from 0~1 to 0~256
                    loc = torch.cat((loc, now_loc), 1)
                    image_features = result['conv']

                text_pred_list_bbox = []
                now_pred = []
                for j in range(max_length):
                    if pred[0][j] != 6737:
                        now_pred.append(pred[0][j])
                    else:
                        break
                text_pred_list_bbox = torch.Tensor(now_pred)[1:].long().cuda()
                pre_text_bbox = ""
                for i in text_pred_list_bbox:
                    if i == (len(alphabet)+2):
                        continue
                    pre_text_bbox += alphabet[i-2]

                # if len(pre_text_bbox) != len(pre_text):
                #     if abs(len(pre_text_bbox) - len(pre_text)) <= 1:
                #         pre_text = self.match_strings(pre_text, pre_text_bbox)
                #         labels = get_labels_from_text(pre_text)

                preds_locs = preds_locs_l_r.clone()
                for n in range(0, 16*2, 2):
                    preds_locs[0][n] = int(loc[0][n//2+2].item()) * ori_lq_w / 256 / 512 # for ocr 32*512
                    preds_locs[0][n+1] = 0 

            print('The predicted text: {}'.format(pre_text))


            if len(pre_text) > 16:
                print('\tToo much characters. The max length is 16.')
                return None

            if len(pre_text) < 1:
                print('\tNo character is detected. Continue...')
                return None

            '''
            Step 3: Generating structure prior.
            '''
            prior_characters = []
            prior_features64 = []
            prior_features32 = []
            labels = torch.Tensor(labels).type(torch.LongTensor).unsqueeze(1)
            try:
                with torch.no_grad():
                    prior_cha, prior_fea64, prior_fea32 = self.modelTSPGAN(styles=w0.repeat(labels.size(0), 1), labels=labels, noise=None)
                prior_characters.append(prior_cha)
                prior_features64.append(prior_fea64)
                prior_features32.append(prior_fea32)
            except:
                traceback.print_exc()
                print('\tError. Continue...')
                return None
            

            '''
            Step 4: Restoring the LR input.
            '''
            with torch.no_grad():   
                sr_results = self.modelSR(LQ, prior_features64, prior_features32, preds_locs)
            sr_results = sr_results * 0.5 + 0.5
            sr_results = sr_results.squeeze(0).permute(1, 2, 0) # .flip(2)
            sr_results = np.clip(sr_results.float().cpu().numpy(), 0, 1) * 255.0

            SR_segment = sr_results[:, :ori_lq_w*4, :]
            SR_list.append(SR_segment)

        ShowSR = self.merge_superresolution_segments(SR_list)
        return ShowSR

    def crop_image_to_segments(self, LQ):
        height, width, _ = LQ.shape
        
        max_width = 32 * 12
        overlap = 64

        segments = []
        current_start = 0
        
        while True:
            current_end = current_start + max_width

            if current_end >= width:
                segment = LQ[:, current_start:width]
                segments.append(segment)
                break
            
            segment = LQ[:, current_start:current_end, :]
            segments.append(segment)

            current_start += (max_width - overlap)
        
        return segments

    def merge_superresolution_segments(self, SR_list):
        overlap_sr = 256
        half_overlap_sr = overlap_sr // 2
        
        merged_image = None
        for i, SR_segment in enumerate(SR_list):
            if i == 0:
                merged_image = SR_segment
            else:
                non_overlap_part = SR_segment[:, half_overlap_sr:, :]
                merged_image = np.concatenate((merged_image[:, :-half_overlap_sr, :], non_overlap_part), axis=1)
        
        return merged_image

    def match_strings(self, a, b):
        len_a = len(a)
        len_b = len(b)

        if len_a > len_b:
            match_after_remove_head = sum(1 for x, y in zip(a[1:], b) if x == y)
            match_after_remove_tail = sum(1 for x, y in zip(a[:-1], b) if x == y)
            
            if match_after_remove_head > match_after_remove_tail:
                return a[1:]
            else:
                return a[:-1]

        elif len_a < len_b:
            match_after_add_head = sum(1 for x, y in zip(b[0] + a, b) if x == y)
            match_after_add_tail = sum(1 for x, y in zip(a + b[-1], b) if x == y)
            
            if match_after_add_head > match_after_add_tail:
                return b[0] + a
            else:
                return a + b[-1]
