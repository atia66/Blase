import os
import numpy as np
import cv2
from Text_detection import imgproc


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls



def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    
    return img_files, mask_files, gt_files



def crop_words_from_craft(image, polys):
    
    crops=[]
    for i, poly in enumerate(polys):
        if poly is None:
            continue
    
        pts = np.array(poly).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
    
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue
    
        crops.append(crop)
    return (crops)

def sort_craft_boxes(boxes, y_threshold_ratio=0.5):
    """
    Sort boxes (from CRAFT) in reading order: top->bottom, left->right.
    Uses robust line grouping based on center y-coordinates.
    """
    if len(boxes) == 0:
        return []

    data = []
    for box in boxes:
        cx = np.mean(box[:, 0])
        cy = np.mean(box[:, 1])
        h = max(box[:, 1]) - min(box[:, 1])
        data.append((box, cx, cy, h))
    
    # Sort boxes top-to-bottom by cy
    data = sorted(data, key=lambda x: x[2])
    
    # Group boxes into lines
    lines = []
    current_line = [data[0]]
    
    for i in range(1, len(data)):
        
        _, _, prev_cy, prev_h = data[i - 1]
        box, cx, cy, h = data[i]
        
        # threshold is proportional to average height
        y_threshold = y_threshold_ratio * ((prev_h+ h)/2)
        
        if abs(cy - prev_cy) <= y_threshold:
            current_line.append(data[i])
        else:
            # Sort current line left-to-right
            current_line.sort(key=lambda x: x[1])
            lines.append(current_line)
            current_line = [data[i]]
    
    # Last line
    current_line.sort(key=lambda x: x[1])
    lines.append(current_line)
    
    # Flatten lines
    sorted_boxes = [b[0] for line in lines for b in line]
    
    return sorted_boxes

