import time
from Text_detection.file_utils import sort_craft_boxes ,crop_words_from_craft

import torch
from torch.autograd import Variable

import cv2
import numpy as np
from Text_detection import craft_utils
from Text_detection import imgproc

from collections import OrderedDict

def copyStateDict(state_dict):
    start_idx = 1 if list(state_dict.keys())[0].startswith("module") else 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(Model, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=False, poly=False,
            canvas_size=1280, mag_ratio=1.5, show_time=False):
    t0 = time.time()


    # Resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = Model(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()


    t0 = time.time() - t0
    t1 = time.time()

    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    render_img = np.hstack((score_text.copy(), score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def run_craft(net,image,text_threshold=0.7,low_text=0.4,link_threshold=0.4,cuda=False,canvas_size=1280,mag_ratio=1.5,poly=False,show_time=False):
    t_start = time.time()

    bboxes, polys, score_text = test_net(
        net, image,
        text_threshold, link_threshold, low_text,
        cuda, poly,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
        show_time=show_time,
    )
    polys = sort_craft_boxes(polys)
    crops=crop_words_from_craft(image, polys)  
    t_end=time.time()
    return crops