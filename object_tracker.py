import os

# comment out below line to enable tensorflow logging outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg as person_det_cfg
from components import config
from components.utils import _jaccard
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
 # defined by tf.keras

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from transform_functions import (
    compute_perspective_transform,
    compute_point_perspective_transformation,
)
from colors import bcolors
import itertools
import imutils
import math
import glob
import yaml
from network.network import SlimModel
from mask_detection import detect_mask

flags.DEFINE_string("framework", "tf", "(tf, tflite, trt")
flags.DEFINE_string("weights", "./checkpoints/yolov4-416", "path to weights file")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_boolean("tiny", False, "yolo or yolo-tiny")
flags.DEFINE_string("model", "yolov4", "yolov3 or yolov4")
flags.DEFINE_string(
    "video", "./data/video/test.mp4", "path to input video or set to 0 for webcam"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string(
    "output_format", "XVID", "codec used in VideoWriter when saving video to file"
)
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_boolean("info", False, "show detailed info of tracked objects")
flags.DEFINE_boolean("count", False, "count objects being tracked on screen")


mask_detection_cfg = config.cfg
min_sizes = mask_detection_cfg['min_sizes']
num_cell = [len(min_sizes[k]) for k in range(len(mask_detection_cfg['steps']))]

mask_model = SlimModel(cfg=mask_detection_cfg, num_cell=num_cell, training=False)
mask_model.load_weights("./checkpoints/weights_epoch_100.h5")
transformed_downoids = None

def get_distance_of_point_from_line(centroid_x, centroid_y, line_x1, line_y1, line_x2, line_y2):
    val = ((centroid_x - line_x1) / (line_x2 - line_x1)) - ((centroid_y - line_y1) / (line_y2 - line_y1))
    return abs(round(val,2))

def get_centroids_and_groundpoints(array_boxes_detected):
    """
    For every bounding box, compute the centroid and the point located on the bottom center of the box
    @ array_boxes_detected : list containing all our bounding boxes
    """
    array_centroids, array_groundpoints = (
        [],
        [],
    )  # Initialize empty centroid and ground point lists
    for index, box in enumerate(array_boxes_detected):
        # Draw the bounding box
        # c
        # Get the both important points
        centroid, ground_point = get_points_from_box(box)
        array_centroids.append(centroid)
        array_groundpoints.append(centroid)
    return array_centroids, array_groundpoints


def get_points_from_box(box):
    """
    Get the center of the bounding and the point "on the ground"
    @ param = box : 2 points representing the bounding box
    @ return = centroid (x1,y1) and ground point (x2,y2)
    """
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    center_x = int(math.ceil((box[0] + box[2]) / 2))
    center_y = int(math.ceil((box[1] + box[3]) / 2))
    # Coordiniate on the point at the bottom center of the box
    center_y_ground = center_y + ((box[3] - box[1]) / 2)
    return (center_x, center_y), (center_x, int(center_y_ground))


def main(_argv):
    
    mask_counter = 0
    no_mask_counter = 0
    
    with open("./config_birdview.yml", "r") as ymlfile:
        bird_view_cfg = yaml.load(ymlfile)

    corner_points = []
    for section in bird_view_cfg:
        corner_points.append(bird_view_cfg["image_parameters"]["p1"])
        corner_points.append(bird_view_cfg["image_parameters"]["p2"])
        corner_points.append(bird_view_cfg["image_parameters"]["p3"])
        corner_points.append(bird_view_cfg["image_parameters"]["p4"])
        img_path = bird_view_cfg["image_parameters"]["img_path"]
        size_height = bird_view_cfg["image_parameters"]["size_height"]
        size_width = bird_view_cfg["image_parameters"]["size_width"]

    tr = np.array(
        [
            bird_view_cfg["image_parameters"]["p4"][0],
            bird_view_cfg["image_parameters"]["p4"][1],
        ]
    )
    tl = np.array(
        [
            bird_view_cfg["image_parameters"]["p2"][0],
            bird_view_cfg["image_parameters"]["p2"][1],
        ]
    )
    br = np.array(
        [
            bird_view_cfg["image_parameters"]["p3"][0],
            bird_view_cfg["image_parameters"]["p3"][1],
        ]
    )
    bl = np.array(
        [
            bird_view_cfg["image_parameters"]["p1"][0],
            bird_view_cfg["image_parameters"]["p1"][1],
        ]
    )
    line_left = np.array(
        [
            bird_view_cfg["image_parameters"]["l1"][0],
            bird_view_cfg["image_parameters"]["l1"][1],
        ]
    )

    line_right = np.array(
        [
            bird_view_cfg["image_parameters"]["l2"][0],
            bird_view_cfg["image_parameters"]["l2"][1],
        ]
    )

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))


    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = "model_data/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == "tflite":
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING]
        )
        infer = saved_model_loaded.signatures["serving_default"]

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    output_video_1 = None

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()

        if return_value:
            image = Image.fromarray(frame)
            
        else:
            print("Video has ended or failed, try a different video format!")
            break
        
        frame_num += 1
        print("Frame #: ", frame_num)
        image = frame.copy()

        matrix, bird_view_img = compute_perspective_transform(
        corner_points, maxWidth, maxHeight, image
        )
        
        height, width, _ = bird_view_img.shape

        transformed_lines = compute_point_perspective_transformation(
                matrix, [line_left,line_right])
        
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == "tflite":
            interpreter.set_tensor(input_details[0]["index"], image_data)
            interpreter.invoke()
            pred = [
                interpreter.get_tensor(output_details[i]["index"])
                for i in range(len(output_details))
            ]
            # run detections using yolov3 if flag is set
            if FLAGS.model == "yolov3" and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(
                    pred[1],
                    pred[0],
                    score_threshold=0.25,
                    input_shape=tf.constant([input_size, input_size]),
                )
            else:
                boxes, pred_conf = filter_boxes(
                    pred[0],
                    pred[1],
                    score_threshold=0.25,
                    input_shape=tf.constant([input_size, input_size]),
                )
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score,
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0 : int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0 : int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0 : int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(person_det_cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #         allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ["person"]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        
        if FLAGS.count:
            cv2.putText(
                frame,
                "Objects being tracked: {}".format(count),
                (5, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 255, 0),
                2,
            )
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(image, bboxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(bboxes, scores, names, features)
        ]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]
        bbox_array = []

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Predictions from mask detection model 
        locs, mask_class, score = detect_mask(image, mask_model)
        if locs:
            locs_array = np.array(locs).reshape(-1,4)
            tf_locs_array = tf.convert_to_tensor(
            locs_array, dtype=tf.float32, dtype_hint=None, name=None
            )

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            bbox_array.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            class_name = track.get_class()
            
            if locs:
                bbox_np_array = np.array([[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]]).reshape(-1,4)
                tf_bbox_np_array = tf.convert_to_tensor(
                bbox_np_array, dtype=tf.float32, dtype_hint=None, name=None
                )

                # Computing IOU between person tracking bbox and mask bounding box  
                jaccard_index = _jaccard(tf_locs_array, tf_bbox_np_array)
                
                # Getting the mask bounding box with largest IOU with person tracking bbox
                idx = jaccard_index.numpy().argmax(axis=0)[0]
                class_name = mask_class[idx]
                loc = locs[idx]
                score_ = score[idx] 
                
                array_centroids, array_groundpoints = get_centroids_and_groundpoints(
                [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]])
                
                new_bbox = compute_point_perspective_transformation(
                matrix, array_centroids
                )                

                if track.bbox and (transformed_lines[0][0] < track.bbox[0][0] < new_bbox[0][0] < transformed_lines[1][0]) and get_distance_of_point_from_line(new_bbox[0][0],new_bbox[0][1],transformed_lines[0][0],transformed_lines[0][1],transformed_lines[1][0],transformed_lines[1][1]) <= 0.1 and not track.counted and track.already_classified:
                  if track.has_mask == "no_mask":
                    no_mask_counter = max(0,no_mask_counter - 1)
                  else:
                    mask_counter = max(0,mask_counter - 1)
                  track.counted = True

                elif class_name == "mask" and score_ >= 0.9 and track.has_mask == "no_mask" and not track.counted:
                    track.has_mask = "mask"    
                    mask_counter += 1
                    if track.already_classified:
                      no_mask_counter = max((0,no_mask_counter - 1))
                    else:
                      track.already_classified = True

                elif track.has_mask == "no_mask" and class_name == "unmask" and not track.already_classified and not track.counted:
                    no_mask_counter += 1
                    track.already_classified = True 

                elif track.has_mask == "no_mask" and class_name == "mask" and score_ < 0.9 and not track.already_classified and not track.counted:
                    no_mask_counter += 1
                    track.already_classified = True

                else:
                    pass

                
                track.bbox = new_bbox             
                
                # Plotting mask detections
                cv2.rectangle(
                    frame,
                    (int(loc[0]), int(loc[1])),
                    (int(loc[2]), int(loc[3])),
                    (0, 0, 255),
                    2,
                    )
                            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            center_x = int(math.ceil((bbox[0]+bbox[2])/2))
            center_y = int(math.ceil((bbox[1]+bbox[3])/2))

            
            # Plotting centroid of person
            cv2.circle(frame,(center_x,center_y),3,(0,255,0),5)

            # Plotting bbox of person
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (
                    int(bbox[2]),
                    int(bbox[3]),
                ),
                (0,255,100),
                3,
            )

            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1] - 30)),
                (
                    int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                    int(bbox[1]),
                ),
                color,
                -1,
            )

            cv2.putText(
                frame,
                track.has_mask,
                (int(bbox[0]), int(bbox[1] - 10)),
                0,
                0.75,
                (255, 255, 255),
                2,
            )

            if FLAGS.info:
                print(
                    "Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                        str(track.track_id),
                        class_name,
                        (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    )
                )

        # Plotting the counting line     
        cv2.line(frame,(int(line_left[0]),int(line_left[1])),(int(line_right[0]),int(line_right[1])),(0,255,0),3)

        cv2.putText(
                frame,
                f"Mask Count:{mask_counter}",
                (20, 70),
                0,
                0.75,
                (255, 255, 255),
                2,
            )

        cv2.putText(
                frame,
                f"No_Mask Count:{no_mask_counter}",
                (20, 90),
                0,
                0.75,
                (255, 255, 255),
                2,
            )

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        # if output flag is set, save video file
        if FLAGS.output:
            if output_video_1 is None:
                fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
                output_video_1 = cv2.VideoWriter(
                    FLAGS.output, fourcc1, 25, (frame.shape[1], frame.shape[0]), True
                )

            elif output_video_1 is not None:
                output_video_1.write(frame)
        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
