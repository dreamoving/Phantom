import os
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
import datetime
import gc

from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UniPCMultistepScheduler

from pipelines.pipeline_pasd import StableDiffusionControlNetPipeline
from myutils.misc import load_dreambooth_lora, rand_name
from myutils.wavelet_color_fix import wavelet_color_fix
from lavis.models import load_model_and_preprocess

# realesrGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

#SwinIR
from SwinIR.models.network_swinir import SwinIR as net

#others
import clip
from synthesis_vqa.vqa_model import VideoQualitySynthesisAnalysis
import myutils.regionmerge as rm

#text
from MARCONet.TextEnhancement import TextRestoration

use_pasd_light = False

if use_pasd_light:
    from models.pasd_light.unet_2d_condition import UNet2DConditionModel
    from models.pasd_light.controlnet import ControlNetModel
else:
    from models.pasd.unet_2d_condition import UNet2DConditionModel
    from models.pasd.controlnet import ControlNetModel


# PASD
pretrained_model_path = "checkpoints/stable-diffusion-v1-5"
ckpt_path = "runs/pasd/checkpoint-100000"
#dreambooth_lora_path = "checkpoints/personalized_models/toonyou_beta3.safetensors"
dreambooth_lora_path = "checkpoints/personalized_models/majicmixRealistic_v7.safetensors"
#dreambooth_lora_path = "checkpoints/personalized_models/Realistic_Vision_V5.1.safetensors"
weight_dtype = torch.float16
device = "cuda"

scheduler = UniPCMultistepScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_model_path}/feature_extractor")
unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
controlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="controlnet")
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
controlnet.requires_grad_(False)

unet, vae, text_encoder = load_dreambooth_lora(unet, vae, text_encoder, dreambooth_lora_path)

text_encoder.to(device, dtype=weight_dtype)
vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
controlnet.to(device, dtype=weight_dtype)

validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )

validation_pipeline._init_tiled_vae(encoder_tile_size=2048, decoder_tile_size=512)

# blip
model, preprocess, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# clip
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
class_list = ['a scene from an old movie', 
              'a logo of brand', 
              'a photo of landscape', 
              'a photo of animation scene or character', 
              'a photo of texts or words', 
              'a photo of a poster with texts', 
              'a photo of animal', 
              'a photo of food', 
              'a photo of goods', 
              'a photo of an object', 
              'a photo of a person', 
              'a photo of several people',
              'a photo of a bill or invoice']
scene_dict = {0: 'movie',
              1: 'logo',
              2: 'landscape',
              3: 'anime',
              4: 'text',
              5: 'text',
              6: 'object',
              7: 'object',
              8: 'object',
              9: 'object',
              10: 'human',
              11: 'human',
              12: 'text'}
text_inputs = clip.tokenize(class_list).to(device)

# text process
text_enhancer = TextRestoration(device=device)

# iqa
mos_weights = os.path.join('synthesis_vqa/weights', 'mos_model_best.pth')
vqa_cfg_file = 'synthesis_vqa/vqa_cfg.json'
vqa_model = VideoQualitySynthesisAnalysis(mos_weights=mos_weights, vqa_cfg_file=vqa_cfg_file)

# GANx4
netscale = 4
model_path = 'realesrgan/weights/RealESRGAN_x4plus.pth'
dni_weight = None
model_GAN = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model_GAN,
    tile=1024,
    tile_pad=10,
    pre_pad=0,
    half=True,
    gpu_id=None)

# GANx2
netscale = 2
model_path = 'realesrgan/weights/RealESRGAN_x2plus.pth'
dni_weight = None
model_GAN_2x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler_2x = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model_GAN_2x,
    tile=1024,
    tile_pad=10,
    pre_pad=0,
    half=True,
    gpu_id=None)

# SwinIR
swin_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
param_key_g = 'params_ema'
pretrained_model = torch.load('SwinIR/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
swin_model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
swin_model.eval()
swin_model = swin_model.to(device)

class params():
    def __init__(self):
        self.init_latent_with_noise = False
        self.offset_noise_scale = 0.0
        self.num_inference_steps = 15
        self.added_noise_level = 400
        self.latent_tiled_size = 384
        self.latent_tiled_overlap = 8

args = params()


def inference(input_image, upscale):
    a_prompt = 'clean, high-resolution, 8k, best quality, masterpiece'
    n_prompt = 'dotted, noise, blur, lowres, oversmooth, longbody, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    prompt = ''
    cfg = 7.5
    seed = random.randint(0, 999999)
    args.num_inference_steps = 15
    merge_flag = True

    current_date = datetime.datetime.now()
    timename = current_date.strftime("%Y%m%d%H%M%S")

    with torch.no_grad():
        seed_everything(seed)
        generator = torch.Generator(device=device)

        input_org_image = input_image.convert('RGB')
        ori_width, ori_height = input_org_image.size
        short_side = min(ori_width, ori_height)
        long_side = max(ori_width, ori_height)

        # iqa
        image_info = {'task_id': 0}
        vqa_result = vqa_model.run(np.array(input_org_image), image_info, vqa_mode='mos')
        mos_score = vqa_result['vqa_mos_info']['mos']

        if (short_side > 1080 or long_side > 2160) and mos_score < 0.55: #resize short side to 1080 if mos score < 0.55
            tmp_scale = min(1080 / short_side, 2160 / long_side)
            input_org_image = input_org_image.resize((round(ori_width*tmp_scale), round(ori_height*tmp_scale)))

            ori_width, ori_height = input_org_image.size
            short_side = min(ori_width, ori_height)
            long_side = max(ori_width, ori_height)

        input_org_image.save(f'output/{timename}_seed{seed}_origin.png')

        image = preprocess["eval"](input_org_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image}, num_captions=1)[0]
        caption = caption.replace("blurry", "clear").replace("noisy", "clean") #
        prompt += f"{caption}" if prompt=="" else f", {caption}"

        prompt = a_prompt if prompt=='' else f"{prompt}, {a_prompt}"
        print(prompt)

        if 1:
            # determine which model to use
            # scene classify
            image_features = clip_model.encode_image(clip_preprocess(input_org_image).unsqueeze(0).to(device))
            text_features = clip_model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, indices = similarity[0].topk(1)
            indice = indices[0].item()

            print(scene_dict[indice], mos_score)

            if short_side > 1080 or long_side > 2160:
                sr_model = 'GAN_2x'
            elif scene_dict[indice] in ['logo', 'text']:
                sr_model = 'SwinIR'
            elif mos_score < 0.55: # for low quality images
                sr_model = 'PASD'
            elif scene_dict[indice] == 'anime':
                if mos_score >= 0.75 or (upscale > 2 and short_side * upscale > 1620):
                    sr_model = 'SwinIR'
                else:
                    sr_model = 'PASD'
            elif scene_dict[indice] == 'landscape':
                if mos_score >= 0.8 or (upscale > 2 and short_side * upscale > 1620):
                    sr_model = 'SwinIR'
                else:
                    sr_model = 'PASD'
            elif scene_dict[indice] in ['movie', 'human']:
                if upscale > 2 and short_side * upscale > 1620:
                    sr_model = 'SwinIR'
                else:
                    sr_model = 'PASD'
            elif scene_dict[indice] == 'object':
                if mos_score >= 0.8 or (upscale > 2 and short_side * upscale > 1620):
                    sr_model = 'GAN'
                else:
                    sr_model = 'PASD'

            print(sr_model)

            if sr_model == 'PASD':
                # determine parameters according to input image size
                if short_side < 120 or mos_score < 0.4:
                    process_size = 384
                elif short_side <= 180:
                    process_size = 512
                else:
                    process_size = 768
                resize_preproc = transforms.Compose([
                    transforms.Resize(process_size, interpolation=transforms.InterpolationMode.BILINEAR),
                ])
                
                if (short_side <= 270 or long_side <= 360) and long_side <= 540:
                    upscale_model = min(upscale, 4)
                elif (short_side <= 360 or long_side <= 480) and long_side <= 720:
                    upscale_model = min(upscale, 3)
                elif (short_side <= 540 or long_side <= 720) and long_side <= 1080:
                    upscale_model = min(upscale, 2)
                else:
                    upscale_model = 1


                rscale = upscale_model
                if mos_score < 0.4:
                    input_image = input_org_image.resize((int(input_org_image.size[0]*0.5), int(input_org_image.size[1]*0.5)))
                    input_image = resize_preproc(input_image)
                else:
                    input_image = input_org_image.resize((input_org_image.size[0]*rscale, input_org_image.size[1]*rscale))
                
                    if min(input_image.size) < process_size:
                        input_image = resize_preproc(input_image)

                input_image = input_image.resize((input_image.size[0]//8*8, input_image.size[1]//8*8))
                width, height = input_image.size
                resize_flag = True

                # determine parameters according to output image size
                out_short_side = min(width, height)
                if out_short_side <= 720:
                    args.init_latent_with_noise = False
                    args.added_noise_level = 200
                elif out_short_side <= 1080:
                    args.init_latent_with_noise = False
                    args.added_noise_level = 400
                else:
                    args.init_latent_with_noise = True
                
                # determine control strength
                if scene_dict[indice] == 'landscape' and mos_score < 0.65:
                    alpha = 0.7
                elif scene_dict[indice] == 'movie' and mos_score < 0.7:
                    alpha = 1.4
                elif scene_dict[indice] == 'anime' and mos_score < 0.65:
                    alpha = 1.25
                elif mos_score < 0.4:
                    alpha = 0.7
                else:
                    alpha = 1.0
                
                try:
                    # PASD
                    image = validation_pipeline(
                            args, prompt, input_image, num_inference_steps=15, generator=generator, height=height, width=width, guidance_scale=cfg, 
                            negative_prompt=n_prompt, conditioning_scale=alpha, eta=0.0,
                        ).images[0]
                    
                    if True: #alpha<1.0:
                        image = wavelet_color_fix(image, input_image)
                
                    if resize_flag: 
                        image = image.resize((ori_width*upscale, ori_height*upscale))

                except Exception as e:
                    print(e)
                    image = Image.new(mode="RGB", size=(ori_width*upscale, ori_height*upscale))
                    merge_flag = False
                gc.collect()
                torch.cuda.empty_cache()

            if sr_model == 'GAN':
                try:
                    # GAN
                    input_image_GAN = np.array(input_org_image)[:, :, ::-1] # convert to numpy BGR
                    image_GAN, _ = upsampler.enhance(input_image_GAN, outscale=upscale)
                    image_GAN = Image.fromarray(image_GAN[:, :, ::-1]) # convert back
                except Exception as e:
                    print(e)
                    image_GAN = Image.new(mode="RGB", size=(ori_width*upscale, ori_height*upscale))
                    merge_flag = False
                gc.collect()
                torch.cuda.empty_cache()

            if sr_model == 'GAN_2x':
                try:
                    # GAN
                    upscale = 2
                    input_image_GAN = np.array(input_org_image)[:, :, ::-1] # convert to numpy BGR
                    image_GAN, _ = upsampler_2x.enhance(input_image_GAN, outscale=upscale)
                    image_GAN = Image.fromarray(image_GAN[:, :, ::-1]) # convert back
                except Exception as e:
                    print(e)
                    image_GAN = Image.new(mode="RGB", size=(ori_width*upscale, ori_height*upscale))
                    merge_flag = False
                gc.collect()
                torch.cuda.empty_cache()

            if sr_model == 'SwinIR':
                try:
                    # SwinIR
                    img_lq = np.array(input_org_image).astype(np.float32) / 255.
                    img_lq = np.transpose(img_lq, (2, 0, 1))
                    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

                    window_size = 8
                    _, _, h_old, w_old = img_lq.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                    image_swin = test(img_lq, swin_model, window_size)
                    image_swin = image_swin[..., :h_old * 4, :w_old * 4]

                    image_swin = image_swin.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    image_swin = np.transpose(image_swin, (1, 2, 0))
                    image_swin = (image_swin * 255.0).round().astype(np.uint8)  # float32 to uint8
                    image_swin = Image.fromarray(image_swin)
                    
                    image_swin = image_swin.resize((ori_width*upscale, ori_height*upscale))
                except Exception as e:
                    print(e)
                    image_swin = Image.new(mode="RGB", size=(ori_width*upscale, ori_height*upscale))
                    merge_flag = False
                gc.collect()
                torch.cuda.empty_cache()

        if sr_model == 'PASD':
            new_image = image
        elif sr_model == 'GAN' or sr_model == 'GAN_2x':
            new_image = image_GAN
        elif sr_model == 'SwinIR':
            new_image = image_swin
        else:
            new_image = image

        # do ocr
        if merge_flag:
            new_image, _, _ = text_enhancer.handle_texts(img=np.array(input_org_image), bg=np.array(new_image), region_merge='meanstd', sf=upscale)
            new_image = Image.fromarray(new_image)

    new_image.save(f'output/{timename}_seed{seed}_res.png')

    return new_image


def test(img_lq, model, window_size):
    # test the image tile by tile
    b, c, h, w = img_lq.size()
    tile = min(768, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    tile_overlap = 32
    sf = 4

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
    output = E.div_(W)

    return output

title = "DreaMoving-Phantom Image Super Resolution and Enhancement"

examples=[['examples/3.png'],['examples/4.png'],['examples/5.png'],['examples/6.png'],
            ['examples/7.png'],['examples/8.png'],['examples/1.png'],['examples/10.png'],
            ['examples/12.png'],['examples/14.png']]
# result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2)


demo = gr.Interface(
    fn=inference, 
    inputs=[gr.Image(type="pil"),
            gr.Slider(label="Upsample Scale", minimum=1, maximum=4, value=2, step=1)
            ],
    outputs=gr.Image(type="pil"),
    title=title,
    examples=examples).queue(concurrency_count=1)

demo.launch(
    server_name="0.0.0.0" if os.getenv('GRADIO_LISTEN', '') != '' else "127.0.0.1",
    share=False,
    root_path=f"/{os.getenv('GRADIO_PROXY_PATH')}" if os.getenv('GRADIO_PROXY_PATH') else ""
)

