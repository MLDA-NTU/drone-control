from djitellopy import Tello
import cv2
import time
from threading import Thread

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
while True:
    img = frame_reader.frame
    if img is None:
        continue
    # Display the resulting frame
    cv2.imshow('frame', img)
    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        tello.land()
        tello.end()
        cv2.destroyAllWindows() 
        break
