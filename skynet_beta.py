"""
 This is the nose tracking code (which we belovingly dub as 'Skynet').
 
 Notes:
  - The pose estimation model currently in use is only capable of tracking 1 subject.
  - The algorithm currently assumes that the nose is always present (for yaw control).
  - There is network delay of approx. 0.5 - 1 second. If the subject constantly changes position back and forth
    with the same period as the delay, resonance may occur (drone becomes rather unstable).
 """

# TODO: implement and tune PID control.
# TODO: implement roll / x-axis control (strafing maneuver; revolve around the subject's head to orient camera to the face).

from interface import Interface
from threading import Thread
import numpy as np
import cv2
from drone_vision.posenet import PoseNet, draw_pose


########## SETTINGS ##########

NOSE_CONF_THRES = 0.2   # Minimum confidence score of the nose keypoint for the yaw controller to take effect.

YAW_KP = 0.15
YAW_KI = 0    # haven't implemented yet
YAW_KD = 0

Z_KP = 0.2
Z_KI = 0      # haven't implemented yet
Z_KD = 0
HEIGHT_SETPOINT_OFFSET = 150    # Positive values bring the setpoint up.

Y_SETPOINT = 180   # Desired vertical distance (pixel) between ear & shoulder keypoints (side with highest score).
Y_CONF_THRES = 0.2   # Minimum confidence score of the keypoints for the y-axis controller to take effect.
Y_KP = 0.2
Y_KI = 0      # haven't implemented yet
Y_KD = 0

###############################

skynet = Interface()
model = PoseNet('drone_vision/posenet_resnet50float_stride16')

cv2.namedWindow('PoseNet Output')
posenet_frame = None

skynet_thread = Thread(target=skynet.run)
skynet_thread.start()

# Wait until drone is ready and starts flying.
while not skynet.tello.is_flying:
    cv2.waitKey(1)

# For nose tracking. Define yaw_u, z_u, y_u here so the program won't crash if the drone is brought to nose tracking
# mode before it has seen a nose.
yaw_u = 40
z_u = 0
y_u = 0
last_yaw_error = 0
last_z_error = 0
last_y_error = 0

# Nose tracking loop.
while True:
    cv2.waitKey(1)

    # Pose estimation.
    posenet_frame = skynet.raw_frame
    keypoints = model.predict_singlepose(posenet_frame)
    draw_pose(posenet_frame, keypoints)

    # For readability.
    FRAME_WIDTH = posenet_frame.shape[1]
    FRAME_HEIGHT = posenet_frame.shape[0]
    current_height = skynet.tello.get_height()

    nose_coord = keypoints[0]["position"]
    nose_conf = keypoints[0]["score"]
    left_ear_coord = keypoints[3]["position"]
    left_ear_conf = keypoints[3]["score"]
    right_ear_coord = keypoints[4]["position"]
    right_ear_conf = keypoints[4]["score"]
    left_shoulder_coord = keypoints[5]["position"]
    left_shoulder_conf = keypoints[5]["score"]
    right_shoulder_coord = keypoints[6]["position"]
    right_shoulder_conf = keypoints[6]["score"]

    # Only apply control if Skynet is not under manual control.
    if not skynet.manual_control:
        
        # Yaw and height control.
        yaw_error = nose_coord[0] - FRAME_WIDTH//2
        z_error = (FRAME_HEIGHT//2 - HEIGHT_SETPOINT_OFFSET) - nose_coord[1]

        if nose_conf >= NOSE_CONF_THRES:
            yaw_u = YAW_KP*yaw_error + YAW_KD*(yaw_error - last_yaw_error)
            z_u = Z_KP*z_error + Z_KD*(z_error - last_z_error)

            last_yaw_error = yaw_error
            last_z_error = z_error
        else:
            # Move towards where the nose was last known.
            if last_yaw_error > 0:
                yaw_u = 40
            elif last_yaw_error < 0:
                yaw_u = -40

            z_u = 0

            # Reset to zero to avoid garbage value when the nose is re-established later.
            last_yaw_error = 0
            last_z_error = 0


        # Pitch control.

        # If both left & right sides have good score, use the average of both.
        # If only one side has good confidence, use that side. If none are good, do not move along y-axis.
        left_bool = left_ear_conf >= Y_CONF_THRES and left_shoulder_conf >= Y_CONF_THRES
        right_bool = right_ear_conf >= Y_CONF_THRES and right_shoulder_conf >= Y_CONF_THRES

        if left_bool or right_bool:
            ear_shoulder_dist = (left_bool*(left_shoulder_coord[1]-left_ear_coord[1]) +
                                right_bool*(right_shoulder_coord[1]-right_ear_coord[1])) / (left_bool + right_bool)
            y_error = Y_SETPOINT - ear_shoulder_dist
            y_u = Y_KP*y_error + Y_KD*(y_error - last_y_error)

            last_y_error = y_error
        else:
            y_u = 0
            last_y_error = 0


        # Clip and apply control values. Also implement deadzone if necessary.
        skynet.yaw_velocity = int(np.clip(yaw_u, -100, 100))
        skynet.z_velocity = int(np.clip(z_u, -30, 30))
        skynet.y_velocity = int(np.clip(y_u, -30, 30))
        skynet.y_velocity = (abs(skynet.y_velocity) > 5)*skynet.y_velocity
        skynet.update()


    # Put some information on the screen.
    text = "yaw_vel: {}".format( str(skynet.yaw_velocity) )
    cv2.putText(posenet_frame, text, (5, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "z_vel: {}".format( str(skynet.z_velocity) )
    cv2.putText(posenet_frame, text, (5, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "y_vel: {}".format( str(skynet.y_velocity) )
    cv2.putText(posenet_frame, text, (5, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "height: {}".format( str(current_height) )
    cv2.putText(posenet_frame, text, (5, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the setpoint.
    cv2.circle(posenet_frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2 - HEIGHT_SETPOINT_OFFSET), 6, (0, 0, 255), -1)

    # Update the frame.
    posenet_frame = cv2.resize(posenet_frame, (480, 360))
    cv2.imshow('PoseNet Output', posenet_frame)