from PIL import Image
import random
import torch
import numpy as np
from einops import repeat
from omegaconf import OmegaConf
import os
from glob import glob

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from test_inpainting import load_state_dict, torch_init_model
from natsort import natsorted

import argparse

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.set_grad_enabled(False)

target_image_size = 512
root_path = "check_points/ref_guided_inpainting"
repeat_sp_token = 50
sp_token = "<special-token>"




def initialize_model(path):
    config = OmegaConf.load(os.path.join(path, "model_config.yaml"))
    model = instantiate_from_config(config.model)
    # repeat_sp_token = config['model']['params']['data_config']['repeat_sp_token']
    # sp_token = config['model']['params']['data_config']['sp_token']

    ckpt_list = glob(os.path.join(path, 'ckpts/epoch=*.ckpt'))
    if len(ckpt_list) > 1:
        resume_path = sorted(ckpt_list, key=lambda x: int(x.split('/')[-1].split('.ckpt')[0].split('=')[-1]))[-1]
    else:
        resume_path = ckpt_list[0]
    print('Load ckpt', resume_path)

    reload_weights = load_state_dict(resume_path, location='cpu')
    torch_init_model(model, reload_weights, key='none')
    if getattr(model, 'save_prompt_only', False):
        pretrained_weights = load_state_dict('pretrained_models/512-inpainting-ema.ckpt', location='cpu')
        torch_init_model(model, pretrained_weights, key='none')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    return sampler


sampler = initialize_model(path=root_path)

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8) # start code目前是純noise
    # TODO: Change here
    # Add a encoder here, and add noise to image, 到我們要的strength
    
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)
        print(batch['image'].shape)
        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples)
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        # TODO: change to sample.decode 現在都是全random強度denoise
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code, 
        )
        # 
        x_samples_ddim = model.decode_first_stage(samples_cfg)
        pred = x_samples_ddim * batch['mask'] + batch['image'] * (1 - batch['mask'])

        result = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)

        result = (result.cpu().numpy().transpose(0, 2, 3, 1) * 255)
        result = result[:, :, 512:]

    return [Image.fromarray(img.astype(np.uint8)) for img in result]
    # return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(source, reference, ddim_steps, num_samples, scale, seed):
    source_img = source["image"].convert("RGB")
    origin_w, origin_h = source_img.size
    ratio = origin_h / origin_w
    init_mask = source["mask"].convert("RGB")
    print('Source...', source_img.size)
    reference_img = reference.convert("RGB")
    print('Reference...', reference_img.size)
    # if min(width, height) > image_size_limit:
    #     if width > height:
    #         init_image = init_image.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.BICUBIC)
    #         init_mask = init_mask.resize((int(width / (height / image_size_limit)), image_size_limit), resample=Image.LINEAR)
    #     else:
    #         init_image = init_image.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.BICUBIC)
    #         init_mask = init_mask.resize((image_size_limit, int(height / (width / image_size_limit))), resample=Image.LINEAR)
    #     init_mask = np.array(init_mask)
    #     init_mask[init_mask > 0] = 255
    #     init_mask = Image.fromarray(init_mask)

    # directly resizing to 512x512
    source_img = source_img.resize((target_image_size, target_image_size), resample=Image.Resampling.BICUBIC)
    reference_img = reference_img.resize((target_image_size, target_image_size), resample=Image.Resampling.BICUBIC)
    init_mask = init_mask.resize((target_image_size, target_image_size), resample=Image.Resampling.BILINEAR)
    init_mask = np.array(init_mask)
    init_mask[init_mask > 0] = 255
    init_mask = Image.fromarray(init_mask)

    source_img = pad_image(source_img)  # resize to integer multiple of 32
    reference_img = pad_image(reference_img)
    mask = pad_image(init_mask)  # resize to integer multiple of 32
    width, height = source_img.size
    width *= 2
    print("Inpainting...", width, height)
    # print("Prompt:", prompt)

    # get inputs
    image = np.concatenate([np.asarray(reference_img), np.asarray(source_img)], axis=1)
    image = Image.fromarray(image)
    mask = np.asarray(mask)
    mask = np.concatenate([np.zeros_like(mask), mask], axis=1)
    mask = Image.fromarray(mask)

    prompt = ""
    for i in range(repeat_sp_token):
        prompt = prompt + sp_token.replace('>', f'{i}> ')
    prompt = prompt.strip()
    print('Prompt:', prompt)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )
    # breakpoint()
    # result = [r.resize((int(512 / ratio), 512), resample=Image.Resampling.BICUBIC) for r in result]
    result = [r.resize((int(origin_w), origin_h), resample=Image.Resampling.BICUBIC) for r in result]
    for r in result:
        print(r.size)

    return result


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Your description here')
    # parser.add_argument('--dataset', type=str, choice=['bear', 'kitchen'])
    # args = parser.parse_args()
    
    # if args.dataset == 'bear':
        # ##### Bear #####
        # ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
        # mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask/"
        # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/train/ours_30000/renders/"
        # ref_img = Image.open("frame_00001.jpg")
        # output_root = "./result_bear1/"
        
    # if args.dataset == 'kitchen':
    ##### Kitchen #####
    ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/images_inpaint_unseen"
    mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/inpaint_object_mask_255_final/"
    source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_incomplete/train/ours_30000/renders"
    ref_img = Image.open("DSCF0656_ori_cleanup.JPG")
    output_root = "./result_kitchen1/"

    
    num_image = len(os.listdir(source_root))
    
    ref_list = natsorted(os.listdir(ref_root))
    source_list = natsorted(os.listdir(source_root))
    mask_list = natsorted(os.listdir(mask_root))
    
    
    
    
    for i in range(1, num_image):
        source_img = Image.open(os.path.join(source_root, source_list[i]))
        mask_img = Image.open(os.path.join(mask_root, mask_list[i]))
        source = {"image": source_img, "mask": mask_img}
        result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=2.5, seed=random.randint(0, 147483647))
        result[0].save(f"{output_root}"+ref_list[i])
        # ref_img = result[0]
        # breakpoint()
    