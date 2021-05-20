import cv2
import numpy as np
import yaml
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, required=True,
	help="path to input video")

args = vars(ap.parse_args())
 
video_path = args["video"]

# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])


vs = cv2.VideoCapture(video_path) 
# Load the frame 
(frame_exists, frame) = vs.read()
cv2.imwrite("./static_frame_from_video.jpg", frame)
vs.release()


# Load the image 

img_path = "./static_frame_from_video.jpg"
img = cv2.imread(img_path)

# Get the size of the image for the calibration
width, height, _ = img.shape

# Create an empty list of points for the coordinates
list_points = list()

# Create a black image and a window
windowName = 'Plane_Transform'
cv2.namedWindow(windowName)

print("Select four points to transform the plane")
# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)
cv2.imshow(windowName, img)


cv2.waitKey(0)
cv2.destroyWindow(windowName)

if len(list_points) == 4:
    cv2.line(img, tuple(list_points[0]), tuple(list_points[1]), (0, 255, 0), 2)
    cv2.line(img, tuple(list_points[1]), tuple(list_points[2]), (0, 255, 0), 2)
    cv2.line(img, tuple(list_points[2]), tuple(list_points[3]), (0, 255, 0), 2)
    cv2.line(img, tuple(list_points[3]), tuple(list_points[0]), (0, 255, 0), 2)

else:
    raise Exception("Select all the four points to proceed")

windowName = 'Line Transform'
cv2.namedWindow(windowName)

print("Select two points to transform the line")
cv2.setMouseCallback(windowName, CallBackFunc)
cv2.imshow(windowName, img)

cv2.waitKey(0)
cv2.destroyWindow(windowName)

if len(list_points) == 6:
    print("Displaying image with all selected points")
    cv2.line(img, tuple(list_points[4]), tuple(list_points[5]), (0, 0, 255), 2)
    cv2.imshow("Final", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    raise Exception("Select all the points for the line to proceed")


if len(list_points) == 6:
            # Return a dict to the YAML file
    config_data = dict(
        image_parameters = dict(
            p2 = list_points[3],
            p1 = list_points[2],
            p4 = list_points[0],
            p3 = list_points[1],
            l1 = list_points[4],
            l2 = list_points[5],
            img_path = img_path,
            size_width=width,
            size_height = height
            ))
            # Write the result to the config file
    with open('./config_birdview.yml', 'w') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)