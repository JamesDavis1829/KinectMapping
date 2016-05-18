from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
import cv2

import ctypes
import _ctypes
import sys

import math


class Kinect:
    # Establish the dimensions of a color frame
    COLOR_DIMENSIONS = (1920, 1080)
    
    # Retrieves a Kinect color frame as soon as it is
    # available, converts it to an image matrix and returns it
    # Resolution: 1080x1920
    def get_color_frame(self):
        # Wait until the Kinect has a new image available
        if self.kinect.has_new_color_frame:

            # Retrieve the last color image from the Kinect
            return self.kinect.get_last_color_frame()

        # If something bad happens, return none
        return None

    # Converts a Kinect color frame to an image openCV can use
    # and returns it in the given dimensions (if specified,
    # otherwise 1920x1080)
    def color_frame_to_cv_img(self,
                              frame,
                              width=COLOR_DIMENSIONS[0],
                              height=COLOR_DIMENSIONS[1]):
        # Reformat it from a vector to an image array
        frame = np.reshape(frame,(Kinect.COLOR_DIMENSIONS[1],Kinect.COLOR_DIMENSIONS[0],4))

        # Remove the meaningless 4th byte from each pixel
        frame = frame[:,:,0:3]

        # Resize image
        return cv2.resize(frame, (width, height))

    # Retrieves a raw Kinect depth frame as soon as it is available
    # Resolution: 512x424, but no formatting is done
    def get_depth_image(self):
        # Wait until the Kinect has a new frame available
        if self.kinect.has_new_depth_frame:

            # Retrieve the last depth frame from the Kinect
            frame = self.kinect.get_last_depth_frame()
            
            return frame
        # If something bad happens, return none
        return None

    # Retrieves a body frame as soon as it becomes available
    def get_body_frame(self):
        if self.kinect.has_new_body_frame(): 
            bodies = self.kinect.get_last_body_frame()
            return bodies
        return None

    # Returns the <x,y,z>  values of each pixel in the color frame
    # Application of results: pointMap[(y,x)] yields the 3D camera
    #                           space position of the color space
    #                           value (y,x). Resolution is 1920x1080.
    def color_to_cam_space_map(self):
        # Get the current depth image
        depthData = self.get_depth_image()

        # Convert depth data to an unsigned short integer array
        depthData = depthData.astype(np.uint16)

        depthCount = depthData.shape[0]

        # Get a pointer to the first index of the depth data
        depthPtr = np.ctypeslib.as_ctypes(depthData)

        # Number of pixels in the camera feed
        colorCount = Kinect.COLOR_DIMENSIONS[0]*Kinect.COLOR_DIMENSIONS[1]

        # Preallocate an array of CameraSpacePoints (one point per pixel)
        mappedPoints = np.ones(colorCount, dtype = np.dtype(PyKinectV2._CameraSpacePoint))

        # Get a pointer to the CameraSpacePoint array
        colorPtr = mappedPoints.ctypes.data_as(ctypes.POINTER(PyKinectV2._CameraSpacePoint))

        # Map all the pixels from the color space to the depth space
        self.kinect._mapper.MapColorFrameToCameraSpace(depthCount, depthPtr, colorCount, colorPtr)

        # Remap pixels so that they correspond to (x,y) pixel locations
        mappedPoints = mappedPoints.reshape((Kinect.COLOR_DIMENSIONS[1],
                                             Kinect.COLOR_DIMENSIONS[0]))

        #Save the result as CSV:
        #subsampled = mappedPoints[1::10]
        #np.savetxt('DepthData.txt', subsampled, fmt='%s,%s,%s')
        #print("Saved!")

        return mappedPoints

    # Takes a point (CameraSpacePoint) from the camera space
    # and maps it to the floor space.
    # A tuple is returned as follows: (x, y, z)
    #   x axis: the line along the floor plane parallel to the image plane
    #   y axis: the line along the floor plane purpendicular to x, parallel to the optical axis
    #   z axis: the normal of the ground plane
    # If the value is not finite, None is returned
    def cam_point_to_floor_space(self, camSpacePoint):
        return  self.floorTransform.point_to_transform_space(camSpacePoint)

    # Calculates the average world point of all the inputted
    # camera points and returns this value
    def get_average_world_point(self, subArray):
        # Get the average point in camera space
        inSpace = self.get_average_camera_point(subArray)

        # Convert from camera to floor space
        onFloor = self.cam_point_to_floor_space(inSpace)

        return onFloor

    # Gets the average point in camera space and returns it
    # mask: The boolean mask of which subarray elements to count and ignore
    #       mask must match the dimensions of subArray. 255 = accept, all others: ignored
    def get_average_camera_point(self, subArray, mask):
        inSpace = np.zeros(3)

        it = np.nditer([subArray, mask])

        for (p, msk) in it:
            if msk == 255:
                # Each element of the subarray is a structured array
                # that needs to be converted to a standard numpy array
                if math.isinf(p['x']) == False:
                    inSpace += np.array([p['x'],p['y'],p['z']])

        # Find the average camera point
        # the np.sum(mask)/255 is the number of masked pixels since
        # elements with the value 255 are accepted and the rest are 0.
        inSpace = inSpace/(np.sum(mask)/255)

        return np.array(inSpace)

    # Gets the first detected floor plane from the body frame, freezing the kinect
    # until it is available.
    def _get_floor_plane_from_body_frame(self):
        bodyFrame = None

        # Flush out bad frames
        while bodyFrame == None:
            bodyFrame = self.get_body_frame()

        # Save floor plane as [a,b,c,d], where ax+by+cz+d=0
        floorPlane = np.array([bodyFrame.floor_clip_plane.x,
                                   bodyFrame.floor_clip_plane.y,
                                   bodyFrame.floor_clip_plane.z,
                                   bodyFrame.floor_clip_plane.w])
		
		# Determine the floor x-axis: [1,0,-a/c]
        floorXaxis = np.array([1,0,-floorPlane[0]/floorPlane[2]])
		
        floorTransform = SpaceTransform(floorPlane, floorXaxis)
		
        return floorTransform
        
    # Automatically connects to the Kinect V2.
	# If autoFloorPlane is true, the Kinect grabs
    # the first available floor clip plane from the Kinect body.
    def __init__(self, autoFloorPlane = True):
        # Get a reference to the Kinect sensor
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth |
                                                      PyKinectV2.FrameSourceTypes_Body)
        if autoFloorPlane:
            self.floorTransform = self._get_floor_plane_from_body_frame()
        else:
            # By default, the floor space is the same as the camera space
            self.floorTransform = SpaceTransform()


# Handles transforming between spaces
class SpaceTransform:
    # All vectors and planes of this class are normalized.
    # All the following np arrays pertain 
    # to the transformed space:
    # 	xy_plane: the normalized xy-plane
    # 	x_axis: vector in x_axis direction
    def __init__(self,
                 xy_plane = np.array([0,0,1,0]),
                 x_axis =  np.array([1,0,0])):

        # Plane origin is where the origin is in the untransformed space
        self.plane_origin = np.array([0,0,0])
        
        self.xy_plane = xy_plane
        self.x_axis =  x_axis/np.linalg.norm(x_axis)

        # Y-axis is automatically calculated from normal and x-axis
        self.y_axis = np.cross(self.xy_plane[0:3], self.x_axis)

    # Creates the transform space from 3 points:
    # origin: the new origin
    # on_x_axis: any point, except the origin, along the x-axis
    # y_pos:  any point in the positive y direction
    def create_from_3_points(self, origin, on_x_axis, y_pos):
        # Set the untranformed origin
        self.plane_origin = origin

        # Find the x-axis vector, normalize it
        self.x_axis = on_x_axis-origin
        self.x_axis = self.x_axis/np.linalg.norm(self.x_axis)
                       
        # Find the normal, normalize it
        n = np.cross(self.x_axis,y_pos-origin)  
        n = n/np.linalg.norm(n)

        # Find the xy plane
        self.xy_plane = np.array([n[0],n[1],n[2], -np.dot(n,origin)])

        #Find the new y-axis
        self.y_axis = np.cross(n, self.x_axis)

    # Transforms a point such that it can be referenced from the
    # coordinate system described by x-axis, in_y, and xy_plane.
    # A 1x3 numpy array is returned, or if the value is not finite,
    # None is returned
    def point_to_transform_space(self, point):
        # extract the norm of the ground plane,
        # The magnitude of normal is already 1 as per kinect docs.
        normal = self.xy_plane[0:3]

        # Find the distance from the point to the plane
        distancePP = np.dot(normal,point)+self.xy_plane[3]

        if (not math.isnan(distancePP)):
            # Find the position closest to the point on the xy plane
            planePos = point - distancePP*normal

            # Distance from the origin to the point's position on the plane 
            posVector = planePos - self.plane_origin

            # Calculate the floor position
            floorPos = np.array([np.dot(self.x_axis, posVector),
                                 np.dot(self.y_axis, posVector),
                                 distancePP])
            return floorPos
        else:
            return None
        
