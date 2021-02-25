"""
 This is the nose tracking code (which we belovingly dub as 'Skynet').
 
 Notes:
  - The pose estimation model currently in use is only capable of tracking 1 subject.
  - There is network delay of approx. 0.5 - 1 second. If the subject constantly changes position back and forth
    with the same period as the delay, resonance may occur (drone becomes rather unstable).
 """

from interface import Interface
from threading import Thread
import numpy as np
import cv2
from drone_vision.posenet import PoseNet, draw_skel_and_kp


########## SETTINGS ##########

# Height PD controller.
Z_KP = 0.2
Z_KD = 0.02
HEIGHT_EAR_CONF_THRES = 0.5    # Minimum confidence threshold of the ear keypoint for the height controller to take effect.
MAX_HEIGHT = 120               # Maximum height read by tello.get_height() before stopping climb.
HEIGHT_SETPOINT_OFFSET = 130   # Positive values bring the setpoint up.

# Pitch PD controller (distance to the head).
Y_SETPOINT = 200     # Desired vertical distance (pixel) between ear & shoulder keypoints (side with highest score).
PITCH_Y_CONF_THRES = 0.5   # Minimum confidence score of the keypoints for the y-axis controller to take effect.
Y_KP = 0.2
Y_KD = 0.2

# Yaw controller and roll control.
NOSE_YAW_KP = 0.13      # PD control for nose setpoint. Preferably slow but stable.
NOSE_YAW_KD = 0.2
EAR_YAW_KP = 0.3        # PID control for ear setpoint. Preferably fast.
EAR_YAW_KI = 0.06
EAR_YAW_KD = 0.15
YR_NOSE_CONF_THRES = 0.2   # Minimum confidence of the nose keypoint for the yaw & roll control to follow it.
ROLL_SPEED = 30
YR_EAR_CONF_THRES = 1.0    # Confidence threshold of the ear keypoint below which yaw & roll control
                           # maneuvers the drone around the head.

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
sum_yaw_error = 0
last_z_error = 0
last_y_error = 0

# Nose tracking loop.
while True:
    cv2.waitKey(1)

    # Pose estimation.
    posenet_frame = skynet.raw_frame
    # get keypoints for single pose estimation. it is a list of 17 keypoints
    pose_scores, keypoint_scores, keypoint_coords = model.estimate_multiple_poses(posenet_frame)

    # For readability.
    FRAME_WIDTH = posenet_frame.shape[1]
    FRAME_HEIGHT = posenet_frame.shape[0]
    current_height = skynet.tello.get_height()

    #take the coords of the first person only
    nose_coord = keypoint_coords[0,0,:]
    nose_conf = keypoint_scores[0,0]
    left_ear_coord = keypoint_coords[0,3,:]
    left_ear_conf = keypoint_scores[0,3]
    right_ear_coord = keypoint_coords[0,4,:]
    right_ear_conf = keypoint_scores[0,4]
    left_shoulder_coord = keypoint_coords[0,5,:]
    left_shoulder_conf = keypoint_scores[0,5]
    right_shoulder_coord = keypoint_coords[0,6,:]
    right_shoulder_conf = keypoint_scores[0,6]

    posenet_frame = draw_skel_and_kp(
        posenet_frame, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.0, min_part_score=0.1)

    # cv2.imshow('posenet', overlay_image)  # show the image with keypoints
    # if cv2.waitKey(1) & 0xFF == ord('q'):  # terminate window when press q
    #     break

    # Only apply control if Skynet is not under manual control.
    if not skynet.manual_control:
        # Height control.

        # If both ears are visible, use the average position of both ears as setpoint.
        # If only one ear is visible, follow that ear.
        # If none is visible, move towards where the setpoint was last known. 
        if not left_ear_conf >= HEIGHT_EAR_CONF_THRES and not right_ear_conf >= HEIGHT_EAR_CONF_THRES:
            if current_height >= MAX_HEIGHT:
                z_u = 0
            elif last_z_error > 0:
                z_u = 50
            elif last_z_error < 0:
                z_u = -50
            
            last_z_error = 0   # Reset to zero to avoid garbage value when the ears are re-established later.

        else:
            if left_ear_conf >= HEIGHT_EAR_CONF_THRES and right_ear_conf >= HEIGHT_EAR_CONF_THRES:
                z_error = (FRAME_HEIGHT//2 - HEIGHT_SETPOINT_OFFSET) - (left_ear_coord[0] + right_ear_coord[0])/2
            elif left_ear_conf >= HEIGHT_EAR_CONF_THRES and right_ear_conf < HEIGHT_EAR_CONF_THRES:
                z_error = (FRAME_HEIGHT//2 - HEIGHT_SETPOINT_OFFSET) - left_ear_coord[0]
            elif left_ear_conf < HEIGHT_EAR_CONF_THRES and right_ear_conf >= HEIGHT_EAR_CONF_THRES:
                z_error = (FRAME_HEIGHT//2 - HEIGHT_SETPOINT_OFFSET) - right_ear_coord[0]

            z_u = Z_KP*z_error + Z_KD*(z_error - last_z_error)
            last_z_error = z_error


        # Pitch control.

        # If both left & right sides have good score, use the average of both.
        # If only one side has good confidence, use that side.
        # If none are good, do not move along y-axis (set y_u to zero).
        left_bool = left_ear_conf >= PITCH_Y_CONF_THRES and left_shoulder_conf >= PITCH_Y_CONF_THRES
        right_bool = right_ear_conf >= PITCH_Y_CONF_THRES and right_shoulder_conf >= PITCH_Y_CONF_THRES

        if left_bool and right_bool:
            ear_shoulder_dist = ((left_shoulder_coord[0]-left_ear_coord[0]) + (right_shoulder_coord[0]-right_ear_coord[0])) / 2
        elif left_bool:
            ear_shoulder_dist = (left_shoulder_coord[0]-left_ear_coord[0])
        elif right_bool:
            ear_shoulder_dist = (right_shoulder_coord[0]-right_ear_coord[0])
        else:
            ear_shoulder_dist = Y_SETPOINT
            last_y_error = 0
            
        y_error = Y_SETPOINT - ear_shoulder_dist
        y_u = Y_KP*y_error + Y_KD*(y_error - last_y_error)

        last_y_error = y_error
            

        # Yaw and roll control (orient the camera towards the face).

        if nose_conf >= YR_NOSE_CONF_THRES:
            # Yaw controller follows the nose.
            yaw_error = nose_coord[1] - FRAME_WIDTH//2
            yaw_u = NOSE_YAW_KP*yaw_error + NOSE_YAW_KD*(yaw_error - last_yaw_error)
            last_yaw_error = yaw_error

            # Roll until both ears are visible.
            if left_ear_conf < YR_EAR_CONF_THRES and right_ear_conf >= YR_EAR_CONF_THRES:
                skynet.x_velocity = ROLL_SPEED
            elif left_ear_conf >= YR_EAR_CONF_THRES and right_ear_conf < YR_EAR_CONF_THRES:
                skynet.x_velocity = -ROLL_SPEED
            else:
                skynet.x_velocity = 0

        # If no subject is visible (no nose and ears), yaw towards where the subject was last known.
        elif left_ear_conf < YR_EAR_CONF_THRES and right_ear_conf < YR_EAR_CONF_THRES:
            if last_yaw_error > 0:
                yaw_u = 30
            elif last_yaw_error < 0:
                yaw_u = -30

            last_yaw_error = 0   # Reset to zero to avoid garbage value when the nose is re-established later.

        # If the nose is not visible but the ear(s) is/are visible...
        else:
            # Left ear has higher confidence: yaw controller follows left ear, roll to the left.
            if left_ear_conf - right_ear_conf > 2:
                skynet.x_velocity = -ROLL_SPEED
                yaw_error = left_ear_coord[1] - FRAME_WIDTH//2
            # Right ear has higher or similar confidence to the left ear: yaw controller follows right ear, roll to the right.
            else:
                skynet.x_velocity = ROLL_SPEED
                yaw_error = right_ear_coord[1] - FRAME_WIDTH//2
            
            yaw_u = EAR_YAW_KP*yaw_error + EAR_YAW_KI*sum_yaw_error + EAR_YAW_KD*(yaw_error - last_yaw_error)
            
            # Windup reset.
            if yaw_error*last_yaw_error < 0:
                sum_yaw_error = 0
            else:
                sum_yaw_error += yaw_error

            last_yaw_error = yaw_error


        # Clip and apply control values. Also implement deadzone if necessary.
        skynet.yaw_velocity = int(np.clip(yaw_u, -100, 100))

        skynet.z_velocity = int(np.clip(z_u, -30, 30))
        skynet.z_velocity = (abs(skynet.z_velocity) > 5)*skynet.z_velocity

        skynet.y_velocity = int(np.clip(y_u, -30, 30))
        skynet.update()


    # Put some information on the screen.
    text = "yaw_vel: {}".format( str(skynet.yaw_velocity) )
    cv2.putText(posenet_frame, text, (5, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "z_vel: {}".format( str(skynet.z_velocity) )
    cv2.putText(posenet_frame, text, (5, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "y_vel: {}".format( str(skynet.y_velocity) )
    cv2.putText(posenet_frame, text, (5, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "x_vel: {}".format( str(skynet.x_velocity) )
    cv2.putText(posenet_frame, text, (5, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text = "ear_conf: {}; {}".format( str(left_ear_conf), str(right_ear_conf) )
    cv2.putText(posenet_frame, text, (5, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the setpoint.
    cv2.circle(posenet_frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2 - HEIGHT_SETPOINT_OFFSET), 6, (0, 0, 255), -1)

    # Update the frame.
    posenet_frame = cv2.resize(posenet_frame, (800, 600))
    cv2.imshow('PoseNet Output', posenet_frame)