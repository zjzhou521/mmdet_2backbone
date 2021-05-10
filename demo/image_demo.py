import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='./demo/demo.jpg')
    parser.add_argument('--config', help='Config file',default='./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='./checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score_thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--async_test',action='store_true',help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    
    # show the results
    ret_img = show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

    # score_thr=args.score_thr,
    # title='result',
    # wait_time=0
    # if hasattr(model, 'module'):
    #     model = model.module
    # ret_img = model.show_result(args.img,result,score_thr=score_thr,show=True,wait_time=wait_time,win_name=title,bbox_color=(72, 101, 241),text_color=(72, 101, 241))

    import cv2
    print("type(ret_img) = ",type(ret_img))
    print("ret_img shape = ",ret_img.shape)
    # ret_img = ret_img[...,[0,1,2]]
    # ret_img = ret_img[...,[0,2,1]]
    # ret_img = ret_img[...,[1,0,2]]
    # ret_img = ret_img[...,[1,2,0]]
    # ret_img = ret_img[...,[2,0,1]]
    # ret_img = ret_img[...,[2,1,0]]
    cv2.imwrite("./demo/pred_demo.jpg",ret_img)



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
