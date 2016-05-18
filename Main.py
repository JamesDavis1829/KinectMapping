# Calculates the position of blobs on a 2D plane using the Kinect camera
# and depth sensor

# Todo: Make plane detection automatic and fool-resistant
#       Make main more user (i.e. programmer) friendly

from CVtoKinect import CVtoKinect
import cv2
import numpy as np
from enum import Enum

# Contains all the possible modes for the program
class GUIMode(Enum):
    Calibration = 0
    Tracking = 1

# The HQ class. Controls the high-end aspects of the
# program such as modes (calibration or tracking) and
# the associated GUIs.
class TrackingHandler:
    # The the program (including the GUI) during a frame of calibration.
    # Returns the next mode for the program.
    def handle_calibration_mode(self):
        scaled_img = self.analyzer.get_img_from_kinect()

        # Be sure to update the color map before tracking in the current frame
        self.analyzer.update_color_to_cam_space_map()

        blob_list = self.analyzer.get_robot_markers(scaled_img)

        for tracked in blob_list:
            pos = self.analyzer.get_world_pos_of_blob(tracked)
            tracked.poly_to_floor_space(self.analyzer)
            print pos,
            
        print "\n"

        cv2.imshow('Detected Points:',scaled_img)
        
        if cv2.waitKey(1) == ord('y'):
            # Find, set, and print the new floor plane
            print self.analyzer.find_floor_plane_from_markers(scaled_img)
            
        return GUIMode.Calibration

    # The the program (including the GUI) during a frame of tracking mode
    # Returns the next mode for the program.     
    def handle_tracking_mode(self):
        pass

    # Passes control of the program to the handler
    def run(self):
        while True:
            # Handle each mode appropriately
            handler = self.modes[self.mode]
            self.mode = handler()

            # Exit the program if q is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
                
    def __init__(self):
        # Set the default mode
        self.mode = GUIMode.Calibration

        # List of all the nodes and their handlers
        self.modes = {
            GUIMode.Calibration: self.handle_calibration_mode,
            GUIMode.Tracking: self.handle_tracking_mode
        }

        # The analyzer controls interface between
        # the kinect and openCV.
        self.analyzer = CVtoKinect()
        
        # Set the precision for screen printing
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

# Start the tracker
k_tracker = TrackingHandler()
k_tracker.run()
