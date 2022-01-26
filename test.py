from mmdet.apis import init_detector, inference_detector

config_file = '/ghome/changjh/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/ghome/changjh/mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = '/ghome/changjh/mmdetection/demo/demo.jpg'
result = inference_detector(model, img)
model.show_result(img, result, out_file='/ghome/changjh/result.jpg')