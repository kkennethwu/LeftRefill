from run_inpaint import predict
from PIL import Image
import random


if __name__ == "__main__":
    # single image inpainting
    ref_img_path = "./frame_00001.jpg"
    source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/train/ours_30000/renders/00001.png"
    ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear_leftrefill/inpaint_2d_unseen_mask/frame_00002.png"
    output_root = "./frame_00002_leftrefill.jpg"
    ref_img = Image.open(ref_img_path)
    
    
    source_img = Image.open(source_root)
    mask_img = Image.open(mask_root)
    
    source = {"image": source_img, "mask": mask_img}
    result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=2.5, seed=random.randint(0, 147483647), eta=1.0) # strength=None(No SD Edit)
    result[0].save(f"{output_root}")
    
    # crop version
    width, height = source_img.size
    left = (width - 512) / 2
    top = (height - 512) / 2
    right = (width + 512) / 2
    bottom = (height + 512) / 2
    crop_area = (left, top, right, bottom)
    
    cropped_source_img = source_img.crop(crop_area); cropped_source_img.save("./tmp/cropped_source_img.jpg")
    cropped_mask_img = mask_img.crop(crop_area); cropped_mask_img.save("./tmp/cropped_mask_img.jpg")
    cropped_ref_img = ref_img.crop(crop_area); cropped_ref_img.save("./tmp/cropped_ref_img.jpg")
    
    source = {"image": cropped_source_img, "mask": cropped_mask_img}
    result = predict(source, cropped_ref_img, ddim_steps=50, num_samples=1, scale=2.5, seed=random.randint(0, 147483647), eta=1.0) # strength=None(No SD Edit)
    result[0].save(f"tmp/{output_root}_cropped.jpg")
    
    
    