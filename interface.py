from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import traceback

# Frames per second of the pygame window display

def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key

class Interface():
    """ Tello display
        Press escape key to quit.
        The controls implemented:
            - T: Takeoff
            - L: Land
            - W, S, A, D: Forward, backward, left and right (x,y)
            - Arrow Keys: Up and down. (z)
            - Q and E: Counter clockwise and clockwise rotations (yaw)     
    """
    def __init__(self, FPS = 120):
        # Init pygame
        pygame.init()
        self.key_dictionary = {'a': pygame.K_a, 's': pygame.K_s, 'd': pygame.K_d, 'w': pygame.K_w,
        'Up': pygame.K_UP, 'Down': pygame.K_DOWN, 'Left': pygame.K_LEFT, 'Right': pygame.K_RIGHT, 
        't': pygame.K_t, 'l': pygame.K_l, 'm': pygame.K_m}

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([480, 320])
        self.key = set()
        self.FPS = FPS

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.y_velocity = 0
        self.x_velocity = 0
        self.z_velocity = 0
        self.yaw_velocity = 0
        self.speed = 50
        self.send_rc_control = False
        self.manual_control = True

        # Ticker implementation
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.FPS)

    def run(self):
        try:
            self.tello.connect()
            self.tello.set_speed(self.speed)

            # In case streaming is on. This happens when we quit this program without the escape key.
            self.tello.streamoff()
            self.tello.streamon()

            frame_read = self.tello.get_frame_read()

        except Exception:
            print("Failed to connect to Tello!\n")
            traceback.print_exc()
            quit()

        stop = False
        prev_time = time.time()
        keymap = {}
        while not stop:
            if(time.time() - prev_time > 1 / self.FPS):
                for event in pygame.event.get():
                    if event.type == pygame.USEREVENT + 1:
                        self.update()
                    elif event.type == pygame.QUIT:
                        stop = True
                    elif event.type == pygame.KEYDOWN:
                        keymap[event.scancode] = event.unicode
                        print('keydown %s pressed' % event.unicode)
                        if event.key == pygame.K_ESCAPE:
                            stop = True
                        else:
                            self.key.add(self.keydown(event.key))
                    elif event.type == pygame.KEYUP:
                        if get_key(self.key_dictionary, event.key) in self.key:
                            self.key.remove(self.keyup(event.key))
                        print('keyup %s pressed' % keymap[event.scancode])

                if frame_read.stopped:
                    break

                self.screen.fill([0, 0, 0])

                self.raw_frame = frame_read.frame
                frame = self.raw_frame.copy()
                
                text = "Battery: {}%".format(self.tello.get_battery())
                cv2.putText(frame, text, (5, 200 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.rot90(frame)
                frame = np.flipud(frame)

                frame = pygame.surfarray.make_surface(frame)
                self.screen.blit(frame, (0, 0))
                pygame.display.update()
                #print("[DEBUG] Key press: {}".format(self.key))
                prev_time = time.time()
                # time.sleep(1 / self.FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update values based on key pressed
        Arguments:
            key: pygame key
        """
        if self.manual_control:
            if key == pygame.K_a:       # set left velocity
                self.x_velocity = -self.speed
            elif key == pygame.K_d:     # set right velocity
                self.x_velocity = self.speed
            elif key == pygame.K_w:     # set forward velocity
                self.y_velocity = self.speed
            elif key == pygame.K_s:     # set backward velocity
                self.y_velocity = -self.speed
            elif key == pygame.K_UP:    # set up velocity
                self.z_velocity = self.speed
            elif key == pygame.K_DOWN:  # set down velocity
                self.z_velocity = -self.speed
            elif key == pygame.K_LEFT:     # set yaw counter clockwise velocity
                self.yaw_velocity = -self.speed
            elif key == pygame.K_RIGHT:     # set yaw clockwise velocity
                self.yaw_velocity = self.speed

        return get_key(self.key_dictionary, key)

    def keyup(self, key):
        """ Update values for key release
        Arguments:
            key: pygame key
        """
        if self.manual_control:
            if key == pygame.K_a or key == pygame.K_d:          # set zero left/right velocity
                self.x_velocity = 0
            elif key == pygame.K_w or key == pygame.K_s:        # set zero forward/backward velocity
                self.y_velocity = 0
            elif key == pygame.K_UP or key == pygame.K_DOWN:    # set zero up/down velocity
                self.z_velocity = 0
            elif key == pygame.K_RIGHT or key == pygame.K_LEFT:        # set zero yaw velocity
                self.yaw_velocity = 0

        if key == pygame.K_t:         # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:         # land
            not self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_m:         # enable/disable manual control (keyboard)
            self.manual_control = not self.manual_control
            self.x_velocity = 0
            self.y_velocity = 0
            self.z_velocity = 0
            self.yaw_velocity = 0

        return get_key(self.key_dictionary, key)

    def update(self):
        """ Update and send all the events to the tello"""
        if self.send_rc_control:
            self.tello.send_rc_control(self.x_velocity, self.y_velocity,
                self.z_velocity, self.yaw_velocity)

def main():
    interface = Interface()
    interface.run()

if __name__ == '__main__':
    main()