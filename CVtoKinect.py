from DiscoverySpotKinect import Kinect

import cv2
import numpy as np
import math

# Manages the interface between the Kinect and OpenCV
class CVtoKinect:
    # Detects all polygons in the given image and returns
    # the ones with the given # of vertices in a python list.
    # perimeter_lim: the min and max allowed polygon perimeters
    # The given image should already be threshholded for best results.
    def detect_polygons(self, image, vertices, perimeter_lim = (50,1000)):
        #Contour mapping overwrites the original image
        original_mask = image.copy()
        
        # The maximum error when maping contours to polygons (higher -> more detail)
        vertex_max_error_scale = 0.03

        # The list of all contours which meet the criteria
        polygons = []
        
        # Find all the contours on the given image
        image, contours, hierarchy = cv2.findContours(image,
                                                      cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)

        # Check the geometry of each contour
        for contour in contours:
            # Find the perimeter of the contour
            perim = cv2.arcLength(contour,True)
            if perim > perimeter_lim[0] and perim < perimeter_lim[1]:
                # Find the polygon approximation of each shape
                approx = cv2.approxPolyDP(contour,perim*vertex_max_error_scale,True)

                # Are there the right number of vertices?
                if len(approx) == vertices:
                    
                    # Add the new blob to the list
                    polygons.append(Blob(contour, original_mask, polyApprox = approx))

        # Restore the original image for posterity
        image = original_mask
        
        return polygons

    # Uses the Kinect library to find the position of the given
    # blob relative to the floor space.
    def get_world_pos_of_blob(self, blob):
        # Find the average floor space point
        camera_space_pos = self.get_camera_position_of_blob(blob)

        # Convert the mean point to floor space
        pos = self.kinectCtrl.cam_point_to_floor_space(camera_space_pos)
        return pos

    # Uses the Kinect library to find the position of the given
    # blob relative to the camera space.
    def get_camera_position_of_blob(self, blob):
         # Scale the blob point for camera-to-depth map
        trueBounds = (self.cvDownscale*np.array(blob.bounds)).astype(np.int32)

        # Make a subarray of all the points in the blob
        blobPoints = self.color_to_cam_space_map[trueBounds[1]:trueBounds[1]+trueBounds[3]:1,
											trueBounds[0]:trueBounds[0]+trueBounds[2]:1]

        # Resize the mask so that there is a one-to-one correspondence
        # between it and the color-to-depth map
        resizedMask = cv2.resize(blob.mask,
                                 (trueBounds[2],trueBounds[3]),
                                 interpolation=cv2.INTER_NEAREST)

        pos = self.kinectCtrl.get_average_camera_point(blobPoints, resizedMask)

        return pos

    # Gets a color image from the Kinect and downscales it
    # according to cvDownscale, then returns it.
    def get_img_from_kinect(self):
        colFrame = self.kinectCtrl.get_color_frame()

        # Rescale the image
        scaled_img =  self.kinectCtrl.color_frame_to_cv_img(colFrame,
                                                               int(Kinect.COLOR_DIMENSIONS[0]/self.cvDownscale),
                                                               int(Kinect.COLOR_DIMENSIONS[1]/self.cvDownscale))
        return scaled_img

    # Detects polygons in a given color range and returns them as
    # an array of keypoints. The input image must be in HSV format.
    def get_colored_polygons(self, hsv_img, min_color, max_color, vertices):
        # Threshold the HSV image for the desired colored pixels
        mask = cv2.inRange(hsv_img, min_color, max_color)

        polys = self.detect_polygons(mask, vertices)

        return polys

    # Attempts to finds the floor plane. Color profiles of the markers are defined
    # in the init function. If the floor plane is identified it is automatically set
    # for the kinect and returned. otherwise, None is returned
    # and the Kinect's previous floor plane is kept.
    # image:  The frame from get_img_from_kinect
    def find_floor_plane_from_markers(self, image):
        # find all the square blobs
        blue_squares, green_squares, red_squares = self.get_floor_plane_markers(image)
        
        # Find the three axes
        origin = self.get_camera_position_of_blob(blue_squares[0])
        y_positive = self.get_camera_position_of_blob(red_squares[0])
        x_positive = self.get_camera_position_of_blob(green_squares[0])

        # Create the floor plane
        self.kinectCtrl.floorTransform.create_from_3_points(origin,x_positive,y_positive)

        return self.kinectCtrl.floorTransform.xy_plane

    # Gets the positions of the square rgb floor plane markers.
    # Three lists are returned, each corresponding to the color scheme in init
    # image: a correctly-scaled bgr image
    def get_floor_plane_markers(self, image):
        # Eliminate high-frequency noise
        blurred = cv2.blur(image, (6,6))
        
        # Convert BGR -> HSV
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # The inverse image makes finding red tones much easier
        # since red occupies both extremes of hue
        hsv_inv_img = cv2.cvtColor(cv2.bitwise_not(blurred), cv2.COLOR_BGR2HSV)

        # Find all the squares by category (squares have 4 vertices)
        origin_squares = self.get_colored_polygons(hsv_img,
                                                 self.origin_hsv_min,
                                                 self.origin_hsv_max, 4)
        
        x_axis_squares = self.get_colored_polygons(hsv_img,
                                                  self.x_axis_hsv_min,
                                                  self.x_axis_hsv_max, 4)
        
        y_pos_squares = self.get_colored_polygons(hsv_inv_img,
                                                  self.p_pos_hsv_min,
                                                  self.p_pos_hsv_max, 4)

        return origin_squares, x_axis_squares, y_pos_squares

    # Detects triangle blobs on top of robots and returns those blobs. No transformations occur.
    def get_robot_markers(self, image):
        # Eliminate high-frequency noise
        blurred = cv2.blur(image, (3,3))
        
        # Convert BGR -> HSV
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Find all the squares by category (squares have 4 vertices)
        triangles = self.get_colored_polygons(hsv_img,
                                              self.robot_hsv_min,
                                              self.robot_hsv_max, 3)

        return triangles

    # Updates self.color_to_cam_space_map with the latest map from KinectCtrl
    def update_color_to_cam_space_map(self):
        self.color_to_cam_space_map = self.kinectCtrl.color_to_cam_space_map()

    def __init__(self):
        # Create a new controller for the Kinect
        self.kinectCtrl = Kinect(False)

        # The HSV ranges for the markers (in a well-lit room)
        
        # Blue
        self.origin_hsv_min = np.array([100,120,60])
        self.origin_hsv_max = np.array([120,255,255])

        #Green
        self.x_axis_hsv_min = np.array([50,70,40])
        self.x_axis_hsv_max = np.array([100,255,255])

        # Red, but for an inverted spectrum
        self.p_pos_hsv_min = np.array([80,80,40])
        self.p_pos_hsv_max = np.array([90,255,255])

        # Green
        self.robot_hsv_min = np.array([50,70,90])
        self.robot_hsv_max = np.array([100,255,255])

        # Defines the ratio of the Kinect frame (1080) over the image
        # that all CV processing is done to. Typically this is 1.5 or more.
        self.cvDownscale = Kinect.COLOR_DIMENSIONS[1]/720.0

        # The color-to-camera space map used by the kinect
        self.color_to_cam_space_map = None

# An alternative to keypoints, for contour-based blobs
class Blob:
    # source_img: the image the contour was found in.
    #   If set, it will copy the part of the subarray
    #   that the contour was found in.
    # contour: the contour itself
    def __init__(self, contour, mask_img=None, polyApprox=None):
        self.contour = contour

        # Format: tuple(x,y,width,height)
        self.bounds = cv2.boundingRect(contour)

        #Save polyApprox, the polygon approximation of the blob, is in case
        # it has a value.
        self.polyApprox = polyApprox
        
        # Splice out the contour region, if mask was given
        if mask_img is not None:
            # Reference the parts area of the image that belong to the blob
            self.mask = mask_img[self.bounds[1]:self.bounds[1]+self.bounds[3]:1,
                                 self.bounds[0]:self.bounds[0]+self.bounds[2]:1]
        else:
            self.mask = None

    # Get the 3d floor positions of each vertex of the approxPoly
    # cv_control: The CVKinect instance for converting the points to floor space
    # color_to_cam_space_map: the map from the color space to the camera space
    def poly_to_floor_space(self, cv_control):
        for vertex in self.polyApprox:
            color_pos = tuple(vertex[0][::-1]*cv_control.cvDownscale)
            
            # Get the camera space positions of the blob
            cam_pos = cv_control.color_to_cam_space_map[color_pos]

            # Convert the coordinates to the right format
            cam_pos = np.array((cam_pos['x'], cam_pos['y'], cam_pos['z']))

            # Convert the coordinates to camera space
            cv_control.kinectCtrl.cam_point_to_floor_space(cam_pos)

##                    # Calculate the area of the polygon
##                    area = cv2.contourArea(contour)
##
##                    # Diameter is calculated as if the polygon was a circle
##                    diameter = math.sqrt(area*4/math.pi)
##                    
##                    # Find the center of the polygon by summing the contour points
##                    location = (np.sum(contour, axis=0)/contour.shape[0])[0]
##
##                    # Convert the new polygon into a keypoint (compatibility with blobs)
##                    keypnt = cv2.KeyPoint(location[0],
##                                          location[1],
##                                          diameter)
