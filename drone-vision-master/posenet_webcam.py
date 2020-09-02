import tensorflow as tf
import numpy as np
import cv2
from djitellopy import Tello
from threading import Thread
from utils import keypoint_decoder, decode_singlepose, draw_keypoints

# SET UP TENSORFLOW LITE
# ----------------------
model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# shape is (batch_size, height, width, channel)
INPUT_HEIGHT = input_details[0]['shape'][1]
INPUT_WIDTH = input_details[0]['shape'][2]
FLOATING_MODEL = input_details[0]['dtype'] == np.float32

INPUT_INDEX = input_details[0]['index']
HEATMAP_INDEX = output_details[0]['index']
OFFSET_INDEX = output_details[1]['index']

#Tello init and takeoff
tello = Tello()
tello.connect()
tello.takeoff()
tello.streamon()

key = None

def controls():
    while True:
        if key == ord('w'):
            tello.move_forward(30)
        elif key == ord('s'):
            tello.move_back(30)
        elif key == ord('a'):
            tello.move_left(30)
        elif key == ord('d'):
            tello.move_right(30)
        elif key == ord('e'):
            tello.rotate_clockwise(30)
        elif key == ord('q'):
            tello.rotate_counter_clockwise(30)
        elif key == ord('r'):
            tello.move_up(30)
        elif key == ord('f'):
            tello.move_down(30)

Thread(target=controls, daemon=True).start()
frame_reader = tello.get_frame_read()

# MAIN LOOP
# ---------
while True:
    img = frame_reader.frame   # read webcam capture
    img_input = cv2.resize(img.copy(), (INPUT_WIDTH, INPUT_HEIGHT)) # resize to fit model's input
    img_input = np.expand_dims(img_input, axis=0)   # add batch dimension

    if FLOATING_MODEL:
        img_input = (np.float32(img_input) - 127.5) / 127.5

    # TensorFlow Lite API
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    interpreter.set_tensor(INPUT_INDEX, img_input)  # load image input to INPUT_INDEX
    interpreter.invoke()    # run the model

    heatmaps = interpreter.get_tensor(HEATMAP_INDEX)    # obtain heatmaps
    offsets = interpreter.get_tensor(OFFSET_INDEX)      # obtain offsets
    heatmaps = np.squeeze(heatmaps) # remove batch dimension
    offsets = np.squeeze(offsets)   # remove batch dimension

    outputStride = 32   # from the model
    keypoints = decode_singlepose(heatmaps, offsets, outputStride)     # list of keypoint, each keypoint is ((y,x), score). see utils.py for implementation

    threshold = 0.2
    scaleX = img.shape[1]/INPUT_WIDTH       # scale back to original image size
    scaleY = img.shape[0]/INPUT_HEIGHT      # scale back to original image size
    draw_keypoints(img, keypoints, threshold=threshold, scaleX=scaleX, scaleY=scaleY)   # see utils.py for implementation

    cv2.imshow("pose", img)     # show the image with keypoints
    key = cv2.waitKey(1) & 0xff
    if key == 27:   # terminate window when press q
        tello.land()
        tello.end()
        break

cv2.destroyAllWindows()