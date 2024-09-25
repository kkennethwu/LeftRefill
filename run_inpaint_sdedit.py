from run_inpaint import LeftRefill
from run_inpaint import GsRender_strength
import os
import argparse


#  #################### Bear ####################
#     #### Stage1: Leftrefill on incomplete + GS Render #####
#     ref_img_path = "./output_scene/bear/00000.png"
#     # source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/train/ours_30000/renders/"
#     source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete_isMasked_3dim_detach_nomeanloss/train/ours_30000_object_removal/renders/"
#     ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
#     mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask_great/"
#     output_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/leftrefill"
#     output_root = "./output_scene/bear/leftrefill"
#     if not os.path.exists(output_root):
#         print("output_root not exist, create one")
#         os.makedirs(output_root)
#     # breakpoint()
        
#     LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root)
#     GsRender(scene="bear")


def bear(strength):
    ref_img_path = "./frame_00001.jpg"
    source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete_isMasked_3dim_detach_nomeanloss/add_unmask_loss_densify/train/ours_10000_object_inpaint/renders/"
    
    ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/images_inpaint_unseen/"
    mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/inpaint_2d_unseen_mask_great/"
    output_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/bear/leftrefill_strength"
    output_root = "./output_scene/bear/leftrefill_"
    
    os.makedirs(output_root +  f'{strength}', exist_ok=True)
    LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root +  f'{strength}', strength)
    
def kitchen(strength):
    ref_img_path = "./DSCF0656_ori_cleanup.JPG"
    source_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/kitchen_incomplete_isMasked_3dim_detach_nomeanloss/add_unmask_loss_densify/train/ours_10000_object_inpaint/renders/"
    
    ref_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/images_inpaint_unseen/"
    mask_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/inpaint_2d_unseen_mask_great/"
    output_root = "/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/data/kitchen/leftrefill_strength"
    output_root = "./output_scene/kitchen/leftrefill_"
    
    os.makedirs(output_root +  f'{strength}', exist_ok=True)
    LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root +  f'{strength}', strength)
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene", type=str, default="kitchen")
    argparser.add_argument("--strength", type=float, default=0.5)
    args = argparser.parse_args()
    
    strength = args.strength
    eval(args.scene)(strength)
    
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
    
    ##### Stage2: Start LeftRefill SDEdit + GS Render #####
    # bear()
    kitchen()
    
    # # for strength in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # for strength in [0.75, 0.5, 0.25]:
    #     # SDEdit
    #     LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength)
    
    #     # run gaussian-splatting w/ strength 0.25
    #     GsRender_strength("bear", strength)
    #     # update leftrefefill source path
    #     source_root = f"/home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_leftrefill_s{strength}/train/ours_40000/renders/"
    # breakpoint()