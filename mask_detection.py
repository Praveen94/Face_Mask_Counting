import cv2
import os
import time

import numpy as np
import tensorflow as tf

from components import config
from components.prior_box import priors_box
from components.utils import decode_bbox_tf, compute_nms, pad_input_image, recover_pad_output, show_image



def parse_predict(predictions, priors, cfg):
    label_classes = cfg['labels_list']

    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg['variances'])

    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]

        score_idx = cls_scores > cfg['score_threshold']

        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, cfg['nms_threshold'], cfg['max_number_keep'])

        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)

        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


def detect_mask(frame,mask_model):
    cfg = config.cfg
    min_sizes = cfg['min_sizes']
    num_cell = [len(min_sizes[k]) for k in range(len(cfg['steps']))]
    img_height, img_width, _ = frame.shape
    img = np.float32(frame.copy())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
    img = img / 255.0 - 0.5

    priors, _ = priors_box(cfg, image_sizes=(img.shape[0], img.shape[1]))
    priors = tf.cast(priors, tf.float32)

    predictions = mask_model.predict(img[np.newaxis, ...])

    boxes, classes, scores = parse_predict(predictions, priors, cfg)
    
    if len(boxes)==0:
      return None,None,None
    
    else:

      boxes = recover_pad_output(boxes, pad_params)

      locs = []
      areas = []
      mask_classes = []
      scores_ = []



      for prior_index in range(len(boxes)):
          x1, y1, x2, y2 = int(boxes[prior_index][0] * img_width), int(boxes[prior_index][1] * img_height),int(boxes[prior_index][2] * img_width), int(boxes[prior_index][3] * img_height)
          locs.append((x1,y1,x2,y2))
          bbox_height = y2 - y1
          bbox_width = x2 - x1
          areas.append(bbox_height * bbox_width)
          scores_.append(scores[prior_index])
          mask_classes.append(cfg['labels_list'][classes[prior_index]])

    return locs,mask_classes,scores_