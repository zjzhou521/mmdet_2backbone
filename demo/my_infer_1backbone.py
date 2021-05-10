import asyncio
from argparse import ArgumentParser
import cv2
import mmcv
import numpy as np
import torch
from mmdet.apis import (async_inference_detector, inference_detector_get_feat, inference_detector, inference_detector_feat,
                        init_detector, show_result_pyplot)
import os
import glob

def get_image_path(dir_name):
    pattern_name = dir_name + '/**/*.[jbptJBPT][pnmiPNMI][gepfGEPF]'
    image_paths=[]
    image_paths.extend(glob.glob(pattern_name,recursive=True))
    pattern_name = dir_name + '/**/*.[jtJT][piPI][efEF][gfGF]'
    image_paths.extend(glob.glob(pattern_name,recursive=True))
    return image_paths
def get_image_name(path):
    i = 0
    name_flag = 0
    image_name = ""
    if(path[0]=='.'): i += 2
    while(i<len(path)):
        if(name_flag!=0):
            image_name += path[i]
        if(path[i]=='/'):
            name_flag = 1
        i += 1
    if('/' in image_name):
        return get_image_name(image_name)
    else:
        return image_name
def divide_image_name(name):
    i = 0
    before = ""
    after = ""
    divide_flag = 0
    while(i<len(name)):
        if(name[i]=='.'):
            divide_flag = 1
        if(divide_flag==0):
            before += name[i]
        else:
            after += name[i]
        i += 1
    return before, after


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--vis_path', help='Image file', default='./data/test_data/vis/')
    parser.add_argument('--ir_path', help='Image file', default='./data/test_data/ir/')
    parser.add_argument('--fuse_path', help='Image file', default='./data/test_data/fuse/')
    parser.add_argument('--feat_fuse_strategy', default='add')
    parser.add_argument('--output_path', default='./data/pred_data/')
    # parser.add_argument('--config', help='Config file',default='./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py')
    # parser.add_argument('--checkpoint', help='Checkpoint file',default='./checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')
    parser.add_argument('--config', help='Config file',default='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score_thr', type=float, default=0, help='bbox score threshold')
    # parser.add_argument('--async_test',action='store_true',help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    os.system("rm "+args.output_path+"*") # clear output data
    # load model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    image_fuse_path_list = get_image_path(args.fuse_path)
    for img_fuse_path in image_fuse_path_list:
        img_fuse_name_before, _ = divide_image_name(get_image_name(img_fuse_path))
        img_vis_path = args.vis_path+"VIS"+img_fuse_name_before+".png"
        img_ir_path = args.ir_path+"IR"+img_fuse_name_before+".png"


        print("img_vis_path = ",img_vis_path)
        # original feature on fused image
        result_fuse_img = inference_detector(model, img_fuse_path) # infer boxes
        # class_names=model.CLASSES

        # only vis
        result_vis = inference_detector(model, img_vis_path) # infer boxes

        # only ir
        result_ir = inference_detector(model, img_ir_path) # infer boxes

        # get feature
        feature_vis = inference_detector_get_feat(model,img_vis_path)
        feature_ir = inference_detector_get_feat(model,img_ir_path)

        # print("type(feature_vis) = ",type(feature_vis)) # <class 'tuple'>
        # print("feature_vis = ",feature_vis)
        # import sys
        # sys.exit()

        # fuse feature with add strategy
        if(args.feat_fuse_strategy=='add'):
            feature_fuse = feature_vis
            for index, everyOne in enumerate(feature_fuse):
                everyOne += feature_ir[index]
            # print("feature_fuse type= ",type(feature_fuse)) # <class 'tuple'>
            # print("feature_fuse = ",feature_fuse)
            # for index, everyOne in enumerate(feature_fuse):
            #     print(index,end=" ")
            #     print("shape = ",everyOne.shape)

        # get pred_img with processed feature
        result_fuse_feat = inference_detector_feat(model,feature_fuse,img_fuse_path)

        # print("result = ",result)
        # print("result_feat = ",result_feat)

        # "imshow_det_bboxes" in "./mmdet/core/visualization/image.py"
        # "show_result" in "./mmdet/models/detectors/base.py"
        # "show_result_pyplot" in "./mmdet/apis/inference.py"
        ret_img_fuse_img = show_result_pyplot(model, img_fuse_path, result_fuse_img, score_thr=args.score_thr) # get masked img (numpy array)
        cv2.imwrite(args.output_path + img_fuse_name_before + "_pred_fuse_img.png",ret_img_fuse_img)
        ret_img_vis = show_result_pyplot(model, img_vis_path, result_vis, score_thr=args.score_thr) # get masked img (numpy array)
        cv2.imwrite(args.output_path + img_fuse_name_before+ "_pred_vis.png",ret_img_vis)
        ret_img_ir = show_result_pyplot(model, img_ir_path, result_ir, score_thr=args.score_thr) # get masked img (numpy array)
        cv2.imwrite(args.output_path + img_fuse_name_before + "_pred_ir.png",ret_img_ir)
        ret_img_fuse_feat = show_result_pyplot(model, img_fuse_path, result_fuse_feat, score_thr=args.score_thr) # get masked img (numpy array)
        cv2.imwrite(args.output_path + img_fuse_name_before + "_pred_fuse_feat.png",ret_img_fuse_feat)





if __name__ == '__main__':
    args = parse_args()
    # if args.async_test:
    #     asyncio.run(async_main(args))
    #     print("here1")
    # else:
    main(args)




# async def async_main(args):
#     # build the model from a config file and a checkpoint file
#     model = init_detector(args.config, args.checkpoint, device=args.device)
#     # test a single image
#     tasks = asyncio.create_task(async_inference_detector(model, args.img))
#     result = await asyncio.gather(tasks)
#     # show the results
#     # show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)
#     score_thr=0.3,
#     title='result',
#     wait_time=0
#     # model.show_result(args.img, result[0], model.CLASSES, out_file='result_{}.jpg'.format(i))
#     model.show_result(img,result,score_thr=score_thr,show=True,wait_time=wait_time,win_name=title,bbox_color=(72, 101, 241),text_color=(72, 101, 241),out_file='result_demo.jpg')
