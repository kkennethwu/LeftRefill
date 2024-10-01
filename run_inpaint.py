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

import subprocess
from tqdm import tqdm



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


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512, strength=None, eta=0.0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8) # start code目前是純noise
    
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)
    
    
    
    
    with torch.no_grad(), torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)
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
        
        # TODO: encode image to latent (w // 8, h // 8), replace start_code with below latent (need to add noise inside sample)
        if strength is not None:
            start_code = model.get_first_stage_encoding(model.encode_first_stage(batch["image"]))

        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code, 
            strength=strength
        )
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

def predict(source, reference, ddim_steps, num_samples, scale, seed, strength=None, eta=0.0):
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
        h=height, w=width,
        strength=strength,
        eta=eta
    )
    

    # result = [r.resize((int(512 / ratio), 512), resample=Image.Resampling.BICUBIC) for r in result]
    result = [r.resize((int(origin_w), origin_h), resample=Image.Resampling.BICUBIC) for r in result]
    for r in result:
        print(r.size)

    return result


def LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength=None):
    ref_img = Image.open(ref_img_path)
    num_image = len(os.listdir(source_root))
    ref_list = natsorted(os.listdir(ref_root))
    source_list = natsorted(os.listdir(source_root))
    mask_list = natsorted(os.listdir(mask_root))
    
    ref_img.save(os.path.join(output_root, ref_list[0]))
    for i in tqdm(range(1, num_image)):
        source_img = Image.open(os.path.join(source_root, source_list[i]))
        mask_img = Image.open(os.path.join(mask_root, mask_list[i]))
        source = {"image": source_img, "mask": mask_img}
        result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=2.5, seed=random.randint(0, 147483647), strength=strength, eta=1.0) # strength=None(No SD Edit)
        
        mask_img_np = np.array(mask_img)
        result_img_np = np.array(result[0])
        source_img_np = np.array(source_img)
        result_img_np[mask_img_np == 0] = source_img_np[mask_img_np == 0]
        result_img = Image.fromarray(result_img_np)
        
        
        result_img.save(os.path.join(output_root, ref_list[i]))
  
def GsRender(scene, port=4455):      
    # run gaussian-splatting w/ strength 0.25
    base_env = os.environ.copy()
    conda_env_path = "/home/kkennethwu/anaconda3/envs/surfel_splatting"
    base_env["PATH"] = f"{conda_env_path}/bin:" + base_env["PATH"]
    base_env["CONDA_PREFIX"] = conda_env_path
    
    _2dgs_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting"
    command = f"""
                python train_fintune.py -s data/{scene}_leftrefill/ \
                -m output/{scene}_leftrefill_lpips --images leftrefill \
                --start_checkpoint /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/{scene}_incomplete/chkpnt30000.pth \
                --iteration 40000 --save_iterations 40000 \
                --checkpoint_iteration 40000 \
                --test_iterations 40000 --lambda_dssim 0.5 \
                --port {port} &&
                python render.py -s data/{scene}_leftrefill/ -m output/{scene}_leftrefill_lpips --skip_mesh"""
    
    
    process = subprocess.Popen(command, shell=True, executable='/bin/bash', env=base_env, cwd=_2dgs_root)
    stdout, stderror = process.communicate()
    print(process.returncode)
    print(stdout)
    print(stderror)
    if process.returncode != 0:
        print("Error in running gaussian-splatting")
        exit(1)
        
def GsRender_strength(scene, strength, port=4455):
    # run gaussian-splatting w/ strength 0.25
    base_env = os.environ.copy()
    conda_env_path = "/home/kkennethwu/anaconda3/envs/surfel_splatting"
    base_env["PATH"] = f"{conda_env_path}/bin:" + base_env["PATH"]
    base_env["CONDA_PREFIX"] = conda_env_path
    
    _2dgs_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting"
    command = f"""
                python train_fintune.py -s data/{scene}_leftrefill/ \
                -m output/{scene}_leftrefill_s{strength}_lpips --images leftrefill \
                --start_checkpoint /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/{scene}_incomplete/chkpnt30000.pth \
                --iteration 40000 --save_iterations 40000 \
                --checkpoint_iteration 40000 \
                --test_iterations 40000  \
                --port {port} && \
                python render.py -s data/{scene}_leftrefill/ -m output/{scene}_leftrefill_s{strength}_lpips --skip_mesh""" 
    process = subprocess.Popen(command, shell=True, executable='/bin/bash', env=base_env, cwd=_2dgs_root)
    stdout, stderror = process.communicate()
    print(process.returncode)
    print(stdout)
    print(stderror)
    if process.returncode != 0:
        print("Error in running gaussian-splatting")
        exit(1)



if __name__ == "__main__":    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', '-d', type=str, default='bear', help='dataset name')
    argparser.add_argument('--scene', '-s', type=str, default='bear', help='scene name')
    argparser.add_argument('--script', type=str, choices=['benchmark', 'ours', 'sdedit'], default='ours', help='script to run')
    argparser.add_argument('--strength', type=float, default=0.5, help='strength for sdedit')
    args = argparser.parse_args()
    
    if args.dataset == 'bear':
        dataset_name = "."
    elif args.dataset == '360':
        dataset_name = "360v2_with_masks"
    elif args.dataset == 'our':
        dataset_name = "our_dataset"
    scene_name = args.scene
    

    if args.script == 'ours':
        #### Stage1: Leftrefill on incomplete + GS Render #####
        ref_img_path = f"./{dataset_name}/{scene_name}/00000.png"
        source_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/images_removal/"
        ref_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/images/"
        mask_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/unseen_mask/"
        # output_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/our_dataset/plant/leftrefill"
        output_root = f"./{dataset_name}/{scene_name}/leftrefill"
        if not os.path.exists(output_root):
            print("output_root not exist, create one")
            os.makedirs(output_root)
        # breakpoint()
            
        LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root)
        os.system(f"cp -r {output_root} /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/")
        
    elif args.script == 'benchmark':
        #### Stage1: Leftrefill on incomplete + GS Render #####
        ref_img_path = f"./{dataset_name}/{scene_name}/00000.png"
        source_root = f"/project/gs-inpainting/data/{dataset_name}/{scene_name}/test_images_rend/"
        ref_root = f"/project/gs-inpainting/data/{dataset_name}/{scene_name}/test_images/"
        mask_root = f"/project/gs-inpainting/data/{dataset_name}/{scene_name}/test_object_masks/"
        output_root = f"/project/gs-inpainting/benchmark/LeftRefill/LeftRefillOnTestView/{dataset_name}/{scene_name}"
        if not os.path.exists(output_root):
            print("output_root not exist, create one")
            os.makedirs(output_root)
        LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root)
   
    elif args.script == 'sdedit':
        strength = args.strength
        ref_img_path = f"./{dataset_name}/{scene_name}/00000.png"
        source_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/{dataset_name}/{scene_name}/exp1/train/ours_10000_object_inpaint/renders/"
        ref_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/images/"
        mask_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/unseen_mask/"
        output_root = f"./home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/{dataset_name}/{scene_name}/leftrefill_{strength}/"
        if not os.path.exists(output_root):
            print("output_root not exist, create one")
            os.makedirs(output_root)
        LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength)
        # GsRender_strength(scene=scene_name, port=4773)
    exit()
    # #################### Bear ####################
    # #### Stage1: Leftrefill on incomplete + GS Render #####
    # ref_img_path = "./output_scene/bear/00000.png"
    # # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/train/ours_30000/renders/"
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000_object_removal/renders/"
    # ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    # mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask_great/"
    # output_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/leftrefill"
    # output_root = "./output_scene/bear/leftrefill"
    # if not os.path.exists(output_root):
    #     print("output_root not exist, create one")
    #     os.makedirs(output_root)
    # # breakpoint()
        
    # LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root)
    # GsRender(scene="bear")
    
    # ##### Stage2: Start LeftRefill SDEdit + GS Render #####``
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_leftrefill_lpips/train/ours_40000/renders/"
    
    # # for strength in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # for strength in [0.75, 0.5, 0.25]:
    #     # SDEdit
    #     LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength)
    
    #     # run gaussian-splatting w/ strength 0.25
    #     GsRender_strength("bear", strength)
    #     # update leftrefefill source path
    #     source_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_leftrefill_s{strength}/train/ours_40000/renders/"
    # breakpoint()
    #################### Kitchen ####################
    ref_img_path = "./output_scene/kitchen/00000.png"
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_incomplete/train/ours_30000/renders"
    source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000_object_removal/renders/"
    ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/images_inpaint_unseen/"
    mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/inpaint_2d_unseen_mask/"
    output_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/leftrefill"
    output_root = "./output_scene/kitchen/leftrefill"
    if not os.path.exists(output_root):
        print("output_root not exist, create one")
        os.makedirs(output_root)
    LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root)
    # GsRender(scene="kitchen", port=4773)
    breakpoint()
    # ##### Stage2: Start LeftRefill SDEdit + GS Render #####``
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_leftrefill_lpips/train/ours_40000/renders/"
    
    # # for strength in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # for strength in [0.75, 0.5, 0.25]:
    #     # SDEdit
    #     LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength)
    
    #     # run gaussian-splatting w/ strength 0.25
    #     GsRender_strength("kitchen", strength, port=4773)
    #     # update leftrefefill source path
    #     source_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_leftrefill_s{strength}/train/ours_40000/renders/"
    # breakpoint()

    
    
    
    
    # if args.dataset == 'bear':
    # ##### Bear #####
    # ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    # mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask/"
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/train/ours_30000/renders/"
    # ref_img = Image.open("frame_00001.jpg")
    # output_root = "./result_bear1/"
        
    # # if args.dataset == 'kitchen':
    # ##### Kitchen #####
    # ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/images_inpaint_unseen"
    # mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/inpaint_object_mask_255_final/"
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_incomplete/train/ours_30000/renders"
    # ref_img = Image.open("DSCF0656_ori_cleanup.JPG")
    # output_root = "./result_kitchen1/"

    # ##### Add strength * noise : Bear #####
    # ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    # mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask/"
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_leftrefill/train/ours_30000/renders/"
    # ref_img = Image.open("frame_00001.jpg")
    # output_root = "./result_bear1_1train/"

    
    # num_image = len(os.listdir(source_root))
    
    # ref_list = natsorted(os.listdir(ref_root))
    # source_list = natsorted(os.listdir(source_root))
    # mask_list = natsorted(os.listdir(mask_root))
    
    
    
    ##### Original Left Refill #####
    ##### Bear #####
    ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask/"
    source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/train/ours_30000/renders/"
    ref_img = Image.open("frame_00001.jpg")
    output_root = "./result_bear1_scale3.5/"
    if not os.path.exists(output_root):
            print("output_root not exist, create one")
            os.makedirs(output_root)  
    
    num_image = len(os.listdir(source_root))
    
    ref_list = natsorted(os.listdir(ref_root))
    source_list = natsorted(os.listdir(source_root))
    mask_list = natsorted(os.listdir(mask_root))
    
    for i in range(1, num_image):
        source_img = Image.open(os.path.join(source_root, source_list[i]))
        mask_img = Image.open(os.path.join(mask_root, mask_list[i]))
        source = {"image": source_img, "mask": mask_img}
        result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=3.5, seed=random.randint(0, 147483647))
        result[0].save(f"{output_root}"+ref_list[i])
        # ref_img = result[0]
        # breakpoint()
        
    # ##### GS Render + LeftRefill #####
    # ##### Add strength * noise : Bear #####
    # ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    # mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask/"
    # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_leftrefill/train/ours_30000/renders/"
    # ref_img = Image.open("frame_00001.jpg")
    # output_root = "./result_bear1_1train/"
    # num_image = len(os.listdir(source_root))
    
    # ref_list = natsorted(os.listdir(ref_root))
    # source_list = natsorted(os.listdir(source_root))
    # mask_list = natsorted(os.listdir(mask_root))
    # for i in range(1, num_image):
    #     source_img = Image.open(os.path.join(source_root, source_list[i]))
    #     mask_img = Image.open(os.path.join(mask_root, mask_list[i]))
    #     source = {"image": source_img, "mask": mask_img}
    #     result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=5.0, seed=random.randint(0, 147483647))
    #     result[0].save(f"{output_root}"+ref_list[i])
     
        
        
    # ##### SD Edit strength #####
    # for strength in [0.02, 0.25, 0.5, 0.75, 0.99]:
    #     ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    #     mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask/"
    #     source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_leftrefill/train/ours_30000/renders/"
    #     ref_img = Image.open("frame_00001.jpg")
    #     output_root = f"./result_bear1_train_strength{strength}/"
    #     if not os.path.exists(output_root):
    #         print("output_root not exist, create one")
    #         os.makedirs(output_root)
        
            
    #     num_image = len(os.listdir(source_root))
    
    #     ref_list = natsorted(os.listdir(ref_root))
    #     source_list = natsorted(os.listdir(source_root))
    #     mask_list = natsorted(os.listdir(mask_root))
    #     for i in range(1, num_image):
    #         source_img = Image.open(os.path.join(source_root, source_list[i]))
    #         mask_img = Image.open(os.path.join(mask_root, mask_list[i]))
    #         source = {"image": source_img, "mask": mask_img}
    #         result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=2.5, seed=random.randint(0, 147483647), strength=strength)
    #         result[0].save(f"{output_root}"+ref_list[i])
    