# Calculate and manage camera calibration
#
import pickle
import cv2
import numpy
from math import tan, sin, cos, atan, atan2, acos, asin, degrees, radians, pi, sqrt, hypot    

class CameraCalibration(object):
    """
    CameraCalibration is a record holding camera calibration data.
    Each RoadImage references a CameraCalibration record, or None (if there is no distortion)
    Usage:
        cal1 = CameraCalibration(mtx=matrix, dist=distortion)
        cal1 = CameraCalibration(objpoints, imgpoints, img_size, flags)
        cal2 = CameraCalibration(cal1)
        cal1.save(<file>)
        cal2 = CameraCalibration(<file>)
    """
    _instances = []   # Keep track of existing instances
    _limit = None     # Set or get limit to access maximum number of instances
    _count = 0        # Count instances
        
    def __new__(cls, *args, **kwargs):
        
        # Search kwargs and initialize arguments
        mtx = kwargs.get('mtx',None)
        dist = kwargs.get('dist',None)
        objpoints = kwargs.get('objpoints',None)
        imgpoints = kwargs.get('imgpoints',None)
        img_size = kwargs.get('img_size',None)
        flags = kwargs.get('flags',0)
        file = kwargs.get('file',None)
        error = None
        size = None
        camheight = None
        ahead = None 
        # Purge known keys
        keys = ['mtx', 'dist', 'objpoints', 'imgpoints', 'img_size', 'flags', 'file']
        keys_to_keep = set(kwargs.keys()) - set(keys)
        kkwargs = {k: kwargs[k] for k in keys_to_keep}
        
        # kwargs contains nothing useful, hence purged kkwargs has the same length
        no_dict = (len(kwargs) == len(kkwargs))
        
        # Check invalid arguments
        if len(kkwargs) > 0: raise TypeError('CameraCalibration: invalid argument %s' % kkwargs.keys())
            
        # Associate unnamed arguments
        # Expected order is objpoints, imgpoints, img_size, mtx, dist (many alternate order will work)
        for arg in args:
            if type(arg) is list: # imgpoints or objpoints
                listdim = numpy.array(arg).ndim
                if listdim == 3:
                    if objpoints is None: objpoints = arg
                elif listdim == 4:
                    if imgpoints is None: imgpoints = arg
                else: raise ValueError('CameraCalibration: invalid argument %s' % str(arg))
            elif type(arg) is tuple and len(arg) == 2 and img_size is None: img_size = arg
            elif type(arg) is numpy.ndarray:  # mtx or dist
                arrshape = arg.shape
                if arrshape == (3,3):
                    if mtx is None:
                        mtx = arg
                        ahead = (mtx[0,2],mtx[1,2],0)  # x,y,z        
                elif arrshape == (1,5):
                    if dist is None: dist = arg
                else: raise ValueError('CameraCalibration: invalid argument %s' % str(arg))
            elif type(arg) is int and flags == 0: flags = arg
            elif type(arg) is CameraCalibration and len(args)==1 and no_dict:
                mtx  = arg.matrix
                dist = arg.distortion
                error= arg.error
                size = arg.size    # Numpy order (height,width)
                ahead = arg.ahead
                camheight = arg.camheight
            elif type(arg) is str and len(args)==1 and no_dict: file = arg 
            else: raise ValueError('CameraCalibration: invalid argument %s' % str(arg))

        # Call cv2 if minimum set of mandatory arguments have been provided
        if objpoints and imgpoints and img_size:
            error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, mtx, dist, flags=flags)
            size = (img_size[1], img_size[0])
            ahead = (mtx[0,2],mtx[1,2],0)
            camheight = None
        elif not (file is None):
            # Read instance from file
            with open(file, 'rb') as f:
                dist_pickle = pickle.load(f)
                mtx = dist_pickle.get('mtx', mtx)
                dist = dist_pickle.get('dist', dist)
                error = dist_pickle.get('error', error)
                size = dist_pickle.get('size', img_size)
                ahead = dist_pickle.get('ahead', ahead)
                camheight = dist_pickle.get('camh', camheight)
        else:
            raise ValueError('CameraCalibration: minimum set of arguments is objpoint, imgpoints and img_size.')

        # Search in _instances
        if mtx.shape==(3,3) and dist.shape==(1,5):
            for cc in cls._instances:
                if (cc.matrix == mtx).all() and (cc.distortion == dist).all(): return cc

        # Can make new one? 
        if not(cls._limit is None):
            if cls._count >= cls._limit: raise RuntimeError('CameraCalibration: too many instances')
        # Make new instance
        obj = super(CameraCalibration, cls).__new__(cls)
        # Last chance to fail. Be tolerant to missing error and img_size values.        
        if (mtx is None) or (dist is None): raise ValueError('CameraCalibration: initialization failed.') 
        cls._count += 1      # Space reservation
        # Initialize instance
        obj.matrix = mtx
        obj.distortion = dist
        obj.error = error
        if size: obj.size = size
        else:    obj.size = img_size
        # ahead direction is initialised at centre of camera
        obj.ahead = ahead 
        obj.camheight = camheight 
        # Reference instance
        cls._instances.append(obj)
        return obj
                
    @classmethod
    def get_limit(cls):
        return cls._limit
        
    @classmethod
    def set_limit(cls,lim):
        if lim is None: cls._limit = None
        else:
            # Set limit cannot be used to delete instances
            if lim < cls._count: raise ValueError('CameraCalibration limit cannot be less than current count.')
            cls._limit = lim
        
    @classmethod
    def get_count(cls):
        return cls._count

    def save(self,file):
        # Save the camera calibration result for later use 
        dist_pickle = {}
        dist_pickle["mtx"] = self.matrix
        dist_pickle["dist"] = self.distortion
        dist_pickle["error"] = self.error
        dist_pickle["size"] = self.size
        dist_pickle["ahead"] = self.ahead
        dist_pickle["camh"] = self.camheight
        pickle.dump( dist_pickle, open(file, 'wb') ) 

    def undistort(self, image):
        assert len(image.shape)==3 or len(image.shape)==2, \
            'CameraCalibration.undistort: image must have 2 or 3 dimensions.' 
        msg = 'CameraCalibration.undistort: image size {0} does not match calibration data {1}.'
        assert self.size and self.size == image.shape[:2], \
            ValueError(msg.format(str(image.shape[:2]),str(self.size)))
        return cv2.undistort(image, self.matrix, self.distortion, None, self.matrix)

    def get_size(self):
        return (self.size[1],self.size[0])  # width x height usual order

    def get_center(self):
        """
        Returns the optical center of the camera. It is not necessarily in the middle,
        for instance if the image has been cropped or if the sensor matrix is off-center.
        The calibration finds it. It is returned as pixel coordinates. By definition,
        it is the straight ahead direction for the camera, and it is the default 
        straight ahead direction for the vehicle.
        """
        return (self.matrix[0,2],self.matrix[1,2])

    def focal_length(self,axis='x'):
        """
        Returns the pixels focal length.
        """
        if axis=='y': return self.matrix[1,1]
        elif axis=='x': return self.matrix[0,0]
        return (self.matrix[0,0],self.matrix[1,1])
    
    def set_ahead(self,x,y,z=0):
        """
        Defines the direction 'straight ahead' of the vehicle as a pixel on the camera image.
        In perspective projections, all the directions in space map to x=X/Z, y=Y/Z. In other
        terms, a parametric line starting from the center of the camera has coordinates
        Z=t, X=x.t Y=y.t, therefore a pixel at (x,y) defines a direction in space.
        The optional parameter z is a distance between the rear axle (the location of the
        car which actually moves forward), and the camera, which moves slightly sideways in
        a turn. This sideway motion is more perceptible when sitting ahead of the front wheels,
        in a bus for instance.
        """
        self.ahead = (x,y,z)

    def get_ahead(self, curvature, zcurvature=None):
        """
        The curvature is the inverse of the radius of curvature.
        This method returns the straight ahead direction based on the curvature of the car
        trajectory, and the ahead direction declared by set_ahead.
        If z is zero (the default), the result is exactly the values passed to set_ahead
        and curvature has no influence.
        zcurvature tells how the terrain ahead curves up (-) or down (+, a hilltop).
        Given the orientation of image axes (y down, x right), a left curvature (left turn)
        has negative sign, and a right curvature has positive sign.
        Note that for positive values of z (usual case for a car with a camera on the windshield)
        the 'ahead' direction will move opposite to the turn.
        """
        centerx,centery = self.get_center()
        aheadx, aheady, aheadz = self.ahead
        focalx, focaly = self.focal_length('xy')

        alpha = atan(aheadz*curvature)
        ahead_x = tan(atan((aheadx - centerx)/focalx) - alpha)*focalx
        if zcurvature:
            beta = atan(aheadz*zcurvature)
            ahead_y = tan(atan((aheady - centery)/focaly) - beta)*focaly
            return ahead_x+centerx, ahead_y+centery
        return ahead_x+centerx, aheady

    @property
    def camera_height(self):
        return self.camheight

    @camera_height.setter
    def camera_height(self,h):
        """
        Define the height of the camera above the ground, in real units (meters).
        """
        self.camheight = h

    def lane_start(self, h=0):
        """
        Returns information about the nearest visible road area, which is visible at the 
        bottom of Road Images, but is usually hidden by the vehicle's hood.
        """
        focalx,focaly = self.focal_length('xy')            # Focal length (pixels)
        centerx,centery = self.get_center()                # Optical axis (pixels)
        aheadx, aheady, aheadz = self.ahead                # Straight ahead direction (pixels)
        width, height = self.get_size()                              # Image dimensions (pixels)
        anglex = atan((aheadx-centerx)/focalx)
        angley = atan((aheady-centery)/focaly)             # ahead versus optical center

        # Notations:
        # d: distance along the camera axis
        # z: distance in the lane direction (same as ahead for the virtual straight lane)
        fov_y = atan((height-centery)/focaly)                   # The lower half of the field of view

        # Technically, the equation giving the distance to the start of (visible) lane
        # must be solved, because h can be dependent on the distance.
        # So it is : d_sol = (self.camheight - h(z_sol)) / tan(fov_y - angley)
        #     and  : z_sol = d_sol / cos(anglex)
        if type(h) is float or type(h) is int:
            # A single number corresponds to h at distance z. h at z=0 and z=z_sol is assumed to be zero
            pass
        elif type(h) is tuple:
            # A tuple of lists ( zlist, hlist ) in real world units
            raise NotImplementedError('CameraCalibration.lane_start: height map h is not implemented.')
        else:
            raise NotImplementedError('CameraCalibration.lane_start: height map h is not implemented.')
        d_sol = self.camheight / tan(fov_y - angley)
        z_sol = d_sol / cos(anglex)       # distance from camera to sol over the ground

        return z_sol, d_sol, anglex, angley
        
    def lane(self, z=70, w=3.7, h=0):
        """
        With default values, it returns the pixel coordinates of a trapeze, which depicts
        a driving lane centered on the camera, starting from the car and extending z real
        units ahead.
        If h is given, instead of placing the camera at camera_height above the trapeze, 
        the lane will be placed camera_height - h *below* the camera. Positive h value
        describe a hill in front of the car, negative ones, a hole. Note that once the 
        car is established in the upward path on the hillside, the hill ahead may appear
        as sloping down. 
        w is the width of the lane. Defaults values for z and w and 70 m and 3.7 m (US std).
        z, w and h can be given as lists. The trapeze will not look like a trapeze in this case.
        The function returns the trapeze as a tuple of 4 (x,y) tuples, the corresponding
        rectangle in real world units (width w, length z) and the distance z_sol between
        the camera and the first visible part of the lane.
        """
        focalx,focaly = self.focal_length('xy')            # Focal length (pixels)
        centerx,centery = self.get_center()                # Optical axis (pixels)
        width, height = self.get_size()                    # Image dimensions (pixels)

        # distance from camera to sol over the ground, direction of ahead as angles
        z_sol, d_sol, anglex, angley = self.lane_start(h)
        
        # Notations:
        # d: distance along the camera axis
        # z: distance in the lane direction (same as ahead for the virtual straight lane)
        z_eol = z
        d_eol = z_eol * cos(anglex)

        if type(h) is float or type(h) is int:
            # A single number corresponds to h at distance z. h at z=0 and z=z_sol is assumed to be zero
            h_eol = h
        elif type(h) is tuple:
            # A tuple of lists ( zlist, hlist ) in real world units
            raise NotImplementedError('CameraCalibration.lane: height map h is not implemented.')
        else:
            raise NotImplementedError('CameraCalibration.lane: height map h is not implemented.')

        # No solving needed here since z is an input.
        angley_end_of_lane = atan((self.camheight-h_eol)/d_eol) + angley # end of lane versus center
        
        # This gives the pixel y coordinate of the far end of the lane, at distance z.
        y_eol = centery + tan(angley_end_of_lane)*focaly

        y_sol = height

        # There is not need to match the car position on the road. The lane is computed as if
        # the camera was centered on it.
        delta_x_sol = z_sol * sin(anglex)
        delta_x_eol = z_eol * sin(anglex)
        w_eol = (w/2)/cos(anglex)
        # On screen real eol positions of lane sides are delta_x_eol +/- w_eol
        x_left_eol = centerx + (delta_x_eol - w_eol) * focalx / d_eol
        x_right_eol = centerx + (delta_x_eol + w_eol) * focalx / d_eol
        x_left_sol = centerx + (delta_x_sol - w_eol) * focalx / d_sol
        x_right_sol = centerx + (delta_x_sol + w_eol) * focalx / d_sol

        # Returns 4 (x,y) corners of trapeze
        trapeze = [[x_left_sol, y_sol], [x_left_eol, y_eol],
                   [x_right_eol, y_eol], [x_right_sol, y_sol]]

        # The bird-eye view of the lane is easier to compute
        # In the general case (anglex != 0), it is a parallelogram
        # The coordinates returned are real world measures (i.e. meters if camheight was in meters)
        # camheight, z, w and h must be expressed in the same units.
        deltaz = w_eol*sin(anglex)
        rectangle = [[-w_eol, z_sol - deltaz], [-w_eol, z_eol - deltaz],
                     [ w_eol, z_eol + deltaz], [ w_eol, z_sol + deltaz]]

        return trapeze, rectangle, z_sol
