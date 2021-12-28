import numpy as np
import torch
import cv2
import torch.backends.cudnn as cudnn
import argparse
import sys
from pathlib import Path
import csv
import os

# yolo import
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# efficientNet import
sys.path.insert(0,'../')
from efficientNet_training.train_efficientnet import TextModel
import json

from functools import reduce
import operator
import math

# PaddleOCR import
sys.path.insert(0,'../PaddleOCR')
import tools.infer.predict_rec as predict_rec
def str2bool(v):
    return v.lower() in ("true", "t", "1")
import re

# PREN import
sys.path.insert(0,'../pren')
from Nets.model import Model
from Utils.utils import *
from Configs.testConf import configs
from recog import Recognizer

num_to_word = json.load(open('../efficientNet_training/new_num_to_word.json', 'r', encoding='utf-8'))

final_output = []

@torch.no_grad()
def run(opt,
        weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Text recognition model
    text_model = TextModel(5038)  # text model
    text_model.load_state_dict(torch.load('../efficientNet_training/final_eff.pt'))
    text_model.to(device)
    text_model.eval()

    # PaddleOCR
    paddle_ocr = predict_rec.TextRecognizer(opt)
    
    # PREN
    checkpoint = torch.load(configs.model_path)
    pren_model = Model(checkpoint['model_config'])
    pren_model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(configs.model_path))

    tester = Recognizer(pren_model)

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    count = 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            img_filename = os.path.basename(p)
            img_name, img_ext = img_filename.split('.')
            # print(img_name)
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # cls_labels = det[:, -1].to(torch.int)
                # for label in cls_labels:
                #     print('class :', label.item())
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                sorted_det = sorted(det, key=lambda s: s[0] + s[1])
                for *xyxy, conf, cls in sorted_det:
                    xyxy = torch.tensor(xyxy)
                    xyxy = xyxy.tolist()
                    conf = torch.tensor(conf)
                    conf = conf.tolist()
                    conf = float(conf)
                    cls = torch.tensor(cls)
                    cls = cls.tolist()
                    cls = int(cls)
                    if cls == 0:
                        if conf > 0.8:
                            temp = []
                            x1, y1, x2, y2 = xyxy
                            x1 = int(x1)
                            x2 = int(x2)
                            y1 = int(y1)
                            y2 = int(y2)
                            img_width = x2 - x1
                            img_height = y2 - y1
                            x3 = int(x1 + img_width)
                            y3 = y1
                            x4 = x1
                            y4 = int(y1 + img_height)
                            temp.append(img_name)
                            temp.append(x1)
                            temp.append(y1)
                            temp.append(x3)
                            temp.append(y3)
                            temp.append(x2)
                            temp.append(y2)
                            temp.append(x4)
                            temp.append(y4)
                            text_img = im0[y1:y2, x1:x2]
                            text_img = cv2.resize(text_img, (64, 64))
                            text_img = torch.from_numpy(text_img.transpose(2, 0, 1)).float()
                            text_img = text_img[None, :, :, :].to(device)
                            pred = text_model(text_img)
                            result = num_to_word[str(pred.argmax(dim=1).item())]
                            temp.append(result)
                            print(temp)
                            final_output.append(temp)
                    
                    # PREN
                    elif cls == 1:
                        if conf > 0.8:
                            x1, y1, x2, y2 = xyxy
                            x1, y1, x2, y2 = xyxy
                            x1 = int(x1)
                            x2 = int(x2)
                            y1 = int(y1)
                            y2 = int(y2)
                            img_width = x2 - x1
                            img_height = y2 - y1
                            x3 = int(x1 + img_width)
                            y3 = y1
                            x4 = x1
                            y4 = int(y1 + img_height)
                            text_img = im0[y1:y2, x1:x2]
                            pren_result = tester.recog(text_img)
                            temp = []
                            pren_result = pren_result
                            pren_result = re.sub(r'[^\w]', ' ', pren_result)
                            pren_result = re.sub(r'\s+', '', pren_result)
                            pren_result = re.sub(r'[0-9]', '!', pren_result)  # check whether has number or not
                            if pren_result.find('!') != -1:
                                rec_res, elapse = paddle_ocr([text_img])
                                for i in range(len(rec_res)):
                                        paddle_content = []
                                        paddle_text = rec_res[i][0]
                                        paddle_text = re.sub(r'[^\w]', ' ', paddle_text)
                                        paddle_text = re.sub(r'\s+', '', paddle_text)
                                        if paddle_text != '':
                                            paddle_content.append(img_name)
                                            paddle_content.append(x1)
                                            paddle_content.append(y1)
                                            paddle_content.append(x3)
                                            paddle_content.append(y3)
                                            paddle_content.append(x2)
                                            paddle_content.append(y2)
                                            paddle_content.append(x4)
                                            paddle_content.append(y4)
                                            paddle_content.append(paddle_text)
                                            print(paddle_content)
                                            final_output.append(paddle_content)
                            else:
                                if pren_result != '':
                                    temp.append(img_name)
                                    temp.append(x1)
                                    temp.append(y1)
                                    temp.append(x3)
                                    temp.append(y3)
                                    temp.append(x2)
                                    temp.append(y2)
                                    temp.append(x4)
                                    temp.append(y4)
                                    temp.append(pren_result)
                                    print(temp)
                                    final_output.append(temp)
                    
                    elif cls == 2:
                        if conf > 1:
                            temp = []
                            x1, y1, x2, y2 = xyxy
                            x1 = int(x1)
                            x2 = int(x2)
                            y1 = int(y1)
                            y2 = int(y2)
                            img_width = x2 - x1
                            img_height = y2 - y1
                            x3 = int(x1 + img_width)
                            y3 = y1
                            x4 = x1
                            y4 = int(y1 + img_height)
                            temp.append(img_name)
                            temp.append(x1)
                            temp.append(y1)
                            temp.append(x3)
                            temp.append(y3)
                            temp.append(x2)
                            temp.append(y2)
                            temp.append(x4)
                            temp.append(y4)
                            no_text = '###'
                            temp.append(no_text)
                            print(temp)
                            final_output.append(temp)

                count += 1

    with open('final_output.csv', 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(final_output)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=str2bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_polygon", type=str2bool, default=True)
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=True)

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(opt, opt.weights, opt.source, opt.imgsz, opt.conf_thres, opt.iou_thres, opt.max_det, opt.device, opt.view_img, opt.save_txt, opt.save_conf, opt.save_crop, opt.nosave, opt.classes, opt.agnostic_nms, opt.augment, opt.visualize, opt.update, opt.project, opt.name, opt.exist_ok, opt.line_thickness, opt.hide_labels, opt.hide_conf, opt.half, opt.dnn)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
