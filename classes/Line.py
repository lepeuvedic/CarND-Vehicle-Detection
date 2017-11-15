import numpy as np
import cv2
from abc import ABC, abstractmethod
from threading import current_thread
import copy

# Module variables
_subclasses = {}

# Abstract Base Class
class Line(ABC):
    """
    Line stores the geometric description of a lane line in real world coordinates.
    The instance can be told to change the reference axes, a functionality which is used to
    update the geometry description when the car moves. The new axes are expressed as a point and a
    direction in the coordinates of the current axes.
    Line can analyze images and fit functions to sets of points in order to initialize or update
    a geometry.
    A new geometry can be blended with an old geometry updated via axes'motion.
    The axes are typically the location and orientation of the car's camera.
    """
    
    # Static variables
    # In each derived class, this class variable is initialised to that class.
    _default_sub = None

    @classmethod
    def Register(cls, subcls, name):
        """
        Module init code calls this to define implementations of Line
        """
        if not issubclass(subcls, cls):
            raise ValueError('Line.Register: type %s must inherit from class Line'% subcls.__name__)
        _subclasses[name] = subcls
        if Line._default_sub is None: Line._default_sub = subcls
        return

    @classmethod
    def Implementations(cls):
        """
        Return a list of implementations available.
        It is necessary to import the module to make the implementations available.
        """
        return list(_subclasses.keys())
    
    @classmethod
    def Get_default(cls):
        """
        Returns the name of the default implementation.
        """
        if Line._default_sub is None: return None
        return Line._default_sub.__name__.split('.')[-1]
    
    @classmethod
    def Set_Default(cls, name):
        """
        Set the default type of instance returned by Factory(name=None).
        """
        if name in _subclasses:
            Line._default_sub = _subclasses[name]
            return
        raise ValueError('Line.Set_default: %s is not a registered implementation of Line.'
                         % name)

    @classmethod
    def Factory(cls, name=None, **kwargs):
        """
        Returns a new instance of subtype 'name'.
        """
        if name is None and not(Line._default_sub is None):
            # Use global default
            return Line._default_sub(**kwargs)
        if name in _subclasses:
            return _subclasses[name](**kwargs)
        if name is None:
            msg = 'Must select a default implementation with Line.Set_default() first!'
        else:
            msg = '%s is not a registered implementation of Line.'
        raise ValueError('Line.Factory: '+msg)

    def __init__(self):
        # Default color is orange-yellow
        self._color = np.array([240, 225, 0, 255], dtype=np.uint8)
        self._blink = 0
        # blink_counter is tested for equality with blink at each call of draw() and incremented
        # if not equal. If equal after incrementation, blink_state flips and blink_counter
        # is reset to zero.
        self._blink_counter = 0
        self._blink_state = True
        self._width = 0.12           # 12 cm wide lines
        # Dict where geometries are stored
        self._geom = {}
        return None

    @property
    def zero(self):
        """
        Returns the key under which the representation of y=0 is stored.
        Usually 'zero', but will be build if required.
        """
        KEY0=('zero',)
        if not(KEY0 in self._geom):
            self.blend(KEY0,key1=('one',),key2=('one',),op='wsum',w1=1,w2=-1)
        return KEY0

    @zero.setter
    def zero(self, val):
        raise ValueError('Line.zero: key for zero is a constant (read-only).')
    
    @property
    def one(self):
        """
        Returns the key under which the representation of y=1 is stored.
        Derived classes shall define y=1 in their __init__ method, and store it under that key.
        """
        return ('one',)

    @one.setter
    def one(self, val):
        raise ValueError('Line.one: key for one is a constant (read-only).')

    def __iter__(self):
        """
        Returns an iterator of the geometries.
        """
        return (K for K in self._geom.keys())
    
    def move(self, key, *, origin, dir, key2=None):
        """
        origin is a tuple, a vector (x,y) from the current axes'origin and the new origin.
        The vector should be estimated based on car speed and direction.
        dir is a unit length vector giving the new "ahead" direction.
        The geometry associated to key is represented in the new axes and
        associated with key2 if supplied.
        """
        # The default implementation assumes that for small changes of origin and direction
        # we can just do nothing.
        if key2: self._geom[key2] = self._geom[key]

    def delete(self, key):
        """
        Delete reference to a geometry. A geometry may be referenced multiple times.
        It will disappear when it is no longer referenced.
        """
        if self.exist(key): del self._geom[key]
        return

    def copy(self, key1, key2):
        """
        Make a deep copy of geometry information under name key2.
        See documentation of module copy to provide deepcopy semantics to user classes.
        """
        g1 = self._geom[key1]
        self._geom[key2] = copy.deepcopy(g1)
        
    @abstractmethod
    def delta(self, key1, key2):
        """
        Assumes that key1 and key2 describe the same geometry with an offset in the origin.
        Returns an estimate of that offset which can be given to 'move' as argument origin.
        """
        pass

    def fit(self, key, x, y, func=None):
        """
        Fit a geometry controlled by additional arguments **kwargs to the points
        given by coordinate arrays x and y. Stores result under 'key'.
        Requires a working blend(op='wsum') in the calling derived class.
        """
        if func is None:
            # The method does not know how to perform the generic data fit. See LinePoly.fit.
            raise ValueError('Line.fit: This method must be called with super() from derived classes.')

        if not(key in self._geom):
            # This parent method is useless when there is no pre-existing geometry
            raise ValueError('Line.fit: This method must be called with super() only when key exists.')

        yref = self.eval(key, z=x)
        meany = np.mean(y)
        # Use not in case we have nans.
        if abs(meany-np.mean(yref)) <= abs(meany):
            safekey = ('Line.fit',current_thread())
            # does 'blend' support op 'wsum' and is key an existing geometry?
            self.blend(safekey, key1=key, key2=key, op='wsum', w1=1, w2=0)
            # It worked: 'wsum' exists and we just backed up key to safekey.
            # normalize y in order to better condition matrix
            y = y - yref
            self.delete(key)
            func(key, x, y, deltay=y)
            self.blend(key, key1=safekey, key2=key, op='wsum', w1=1, w2=1)
            self.delete(safekey)
        else:
            self.delete(key)
            func(key, x, y, deltay=y - yref)
        return

    def blend(self, key, *, key1, key2, **kwargs):
        """
        Takes geometric information associated to key1 and key2 and blends it into a new geometric
        information which is stored under key. All keys must be hashables (e.g. strings). 
        For instance key1 can be an old estimate which has been updated using 'move', and 
        key2 can be a new estimate based on an image from the current location.
        """
        # Default implementation does not use key1
        self._geom[key] = self._geom[key2]

    @abstractmethod
    def eval(self, key, *, z, der=0):
        """
        Computes real world x coordinates associated to z coordinates supplied as a numpy array.
        z coordinates are distances from the camera.
        """
        raise NotImplementedError('Line.eval: Class Line is not intended to be used directly.')

    @abstractmethod
    def tangent(self, key, *, z, order=1):
        """
        Simplifies key to a geomtry of order=order, which is tangent to key at z=z.
        """
        raise NotImplementedError('Line.tangent: Class Line is not intended to be used directly.')
    
    def curvature(self, key, *, z):
        """
        Returns the signed curvature. Positive means right turn, negative left turn, zero straight.
        The radius of curvature is the inverse of the curvature.
        """
        return self.eval(key, z=z, der=2)/abs(1+self.eval(key, z=z, der=1)**2)**1.5

    def stats(self, key, *args):
        """
        Returns a tuple. A KeyError exception is thrown if key is not found. Any value
        can be passed: lines.stats( key, 'dens', 'zmax', 'toto', 'cov' ) --> dens , zmax, None, V
        """
        def value_or_None(d,k):
            try:
                return d[k]
            except KeyError:
                return None
            
        l = self._geom[key]
        if len(args)==1:
            return value_or_None(l,args[0])
        return tuple([ value_or_None(l,k) for k in args])

    def set(self, key, name, val):
        """
        Stores arbitrary data with the line.
        """
        if name == 'poly':
            raise ValueError('Line.set: Cannot set poly.')
        self._geom[key][name] = val
        return

    def unset(self, key, name):
        """
        Removes a piece of caller owned data.
        """
        if name == 'poly':
            raise ValueError('Line.unset: Cannot unset poly.')
        del self._geom[key][name]
        return
    
    @property
    def color(self):
        """
        Returns the current line color as a numpy array with 4 integer values between 0 and 255.
        """
        return self._color

    @classmethod
    def _normalize_color(cls, color, nb_ch=4):
        """
        Be flexible regarding color format, but always store as numpy array of uint8.
        Optionally truncate to a given number of channels.
        """
        color = np.array(list(color))
        if np.can_cast(color, np.uint8, casting='safe'): # dtype-based comparison
            color = color.astype(np.uint8)
        elif np.can_cast(color, np.int64, casting='safe'):
            if all([np.can_cast(c, np.uint8, casting='safe') for c in color]):
                color = color.astype(np.uint8)
            else:
                raise ValueError('Line.color: integer RGBA color values must be in [0, 255].')
        elif np.can_cast(color, np.float64, casting='safe'):
            # Must be float between 0 and 1
            if color.min() < 0. or color.max() > 1.:
                raise ValueError('Line.color: float RGBA color values must be in [0., 1.].')
            color = np.round(color*255.).astype(np.uint8)
        else:
            raise ValueError('Line.color: invalid color %s. Use RGB representation as list or numpy array.'%repr(color))
        return color[:nb_ch].tolist()
    
    @color.setter
    def color(self, color):
        """
        Defines the color associated to this Line.
        color is an RGBA tuple, a list or a numpy array with 4 values. 
        blink is going to make the line blink on successive calls to draw.
        With a value of 0, the line is not blinking.
        """
        self._color = Line._normalize_color(color)
        return

    @property
    def blink(self):
        return self._blink
    
    @blink.setter
    def blink(self, val):
        self._blink = val
        if self._blink_counter >= val:
            # We suddendly lowered blink and must avoid freezing the blink_state and
            # the blink_counter. Setup to flip at next draw, and ensure state is True if blinking stops.
            self._blink_counter = val-1
            if val == 0: self._blink_state=False
        return

    @property
    def width(self):
        """
        Returns the width of the painted lane line in world units.
        """
        return self._width

    @width.setter
    def width(self, val):
        """
        width must be >0 and more than 1 pixel (but that cannot be checked here)
        """
        if val <= 0:
            raise ValueError('Line.width: width must be strictly positive.')
        self._width = val

    def exist(self, *keys):
        """
        Return True if all the keys passed as arguments exist, or else return False.
        """
        for k in keys:
            if not(issubclass(type(k), tuple)):
                raise ValueError('Line.exist: keys must be tuples suitable as dictionary keys.')
            if not(k in self._geom): return False
        return True
        
    def draw(self, key, image, *, origin, scale, color=None, width=None, warp=None, unwarp=None):
        """
        Draws a smooth graphical representation of the lane line in an image, taking into 
        account origin and scale. }
        If image is not warped, the line is drawn in a warped buffer, then unwarped and alpha
        blended into the image.
        Returns None: the line is drawn in image.
        """
        try:
            # retrieve 'dens'
            density = self._geom[key]['dens']
        except KeyError:
            density = 1
            
        # Enable blinking line if there are dashes
        # Callers can test blinking.
        if density < 0.6:
            self.blink = 9
        
        # Blink processing
        if self._blink_counter < self._blink:
            self._blink_counter += 1
            if self._blink_counter == self._blink:
                self._blink_state = not(self._blink_state)
                self._blink_counter = 0
                
        # Create buffer
        # Make a fresh road image buffer from image
        buffer = np.zeros_like(image, subok=True)
        if not(image.warped):
            if warp is None or unwarp is None:
                raise ValueError('Line.draw: warp and unwarp function handles must be provided to work on unwarped image.')
            # Create a warped buffer
            buffer = warp(buffer)

        height, _, nb_ch = buffer.shape

        sx,sy = scale
        try:
            # retrieve 'zmax'
            z_max = self._geom[key]['zmax']
        except KeyError:
            z_max = np.inf
        z_max = min(z_max, sy*height)
        
        # Color and width processing (color is stored/normalized as values in [0,255])
        if color is None: color = self.color
        else:             color = Line._normalize_color(color)
        if len(color)>=4:   alphaval = color[3]
        else:               alphaval = 255
        if image.dtype==np.float32 or image.dtype==np.float64:
            color = [ c/255 for c in color]
        
        if width is None: width = self.width
        
        # Call eval with key to get lane line skeleton in world coordinates
        x0,y0 = origin     # (out of image) pixel coordinates of camera location

        rng = range(0, int(z_max/sy), int(2./sy))  # one segment every 2 meters
        z = sy * np.array([(y0-height)+y for y in rng], dtype=np.float32)
        x = self.eval(key,z=z)
        
        # Compute pixel coordinates using sx,sy and origin
        x = x0 + x/sx
        y = y0 - z/sy
            
        # Plot as polygon
        # fillConvexPoly is able to handle any poly which crosses at most 2 times each scan line
        # and does not self-intersect. Ours is a function and fits the definiion.
        # We work with antialiasing and 3 bits of subpixels: every coordinate is multplied by 8
        shift=3
        thick = int(2**shift * width / sx / 2.)
        points_up = [ [int(2**shift * xi - thick), int(2**shift * yi)] for xi,yi in zip(x,y) ]
        points_dn = [ [int(2**shift * xi + thick), int(2**shift * yi)] for xi,yi in zip(x,y) ]
        points_dn.reverse()
        pts = np.array(points_up + points_dn, dtype=np.int32)

        cv2.fillConvexPoly(buffer, pts, color=color[:nb_ch], shift=shift, lineType=cv2.LINE_AA)
        if nb_ch == 4:
            # image, buffer have depth 4
            # Alpha blend buffer
            # buffer contains RGBA data, with per pixel alpha
            alpha = buffer.channel(3)
        else:
            # Draw alpha mask
            alpha = np.zeros_like(buffer.channel(0), dtype=np.uint8, subok=True)
            cv2.fillConvexPoly(alpha, pts, color=[alphaval], shift=shift, lineType=cv2.LINE_AA)
        alpha.binary=False
        alpha = alpha.to_float()

        if not(image.warped):
            buffer = unwarp(buffer)
            alpha = unwarp(alpha)

        if image.dtype == np.float32 or image.dtype == np.float64:
            image *= (1-alpha)
            image += (alpha*buffer)
        else:
            # Integers
            image[:] = (image*(1-alpha)).astype(np.uint8)
            image +=   (alpha * buffer).astype(np.uint8)
        return None

    def draw_area(self, key1, key2, image, *, origin, scale, color=None, warp=None, unwarp=None):
        """
        Draws a smooth graphical representation of the lane line in an image, taking into 
        account origin and scale.
        If image is not warped, the line is drawn in a warped buffer, then unwarped and alpha
        blended into the image.
        Returns None: the line is drawn in image.
        """
        # Blink processing
        if self._blink_counter < self._blink:
            self._blink_counter += 1
            if self._blink_counter == self._blink:
                self._blink_state = not(self._blink_state)
                self._blink_counter = 0
                
        # Create buffer
        # Make a fresh road image buffer from image
        buffer = np.zeros_like(image, subok=True)
        if not(image.warped):
            if warp is None or unwarp is None:
                raise ValueError('Line.draw_area: warp and unwarp function handles '+
                                 'must be provided to work on unwarped image.')
            # Create a warped buffer
            buffer = warp(buffer)

        height, _, nb_ch = buffer.shape
        sx,sy = scale
        try:
            # retrieve 'zmax'
            z_max = min(self._geom[key1]['zmax'],self._geom[key2]['zmax'])
        except KeyError:
            z_max = sy * height
            
        # Color processing
        if color is None: color = self.color
        else:             color = Line._normalize_color(color)
        if len(color)>=4:   alphaval = color[3]
        else:               alphaval = 255
        if image.dtype==np.float32 or image.dtype==np.float64:
            color = [ c/255 for c in color]
        
        # Call eval with key to get lane line skeleton in world coordinates
        x0,y0 = origin     # (out of image) pixel coordinates of camera location

        rng = range(0, int(z_max/sy), int(2./sy))  # one segment every 2 meters
        z = sy * np.array([(y0-height)+y for y in rng], dtype=np.float32)
        x1 = self.eval(key1,z=z)
        x2 = self.eval(key2,z=z)
        
        # Compute pixel coordinates using sx,sy and origin
        x1 = x0 + x1/sx
        x2 = x0 + x2/sx
        y = y0 - z/sy
            
        # Plot as polygon
        # fillConvexPoly is able to handle any poly which crosses at most 2 times each scan line
        # and does not self-intersect. Ours is a function and fits the definiion.
        # We work with antialiasing and 3 bits of subpixels: every coordinate is multplied by 8
        shift=3
        points_up = [ [int(2**shift * xi), int(2**shift * yi)] for xi,yi in zip(x1,y) ]
        points_dn = [ [int(2**shift * xi), int(2**shift * yi)] for xi,yi in zip(x2,y) ]
        points_dn.reverse()
        pts = np.array(points_up + points_dn, dtype=np.int32)

        cv2.fillConvexPoly(buffer, pts, color=color[:nb_ch], shift=shift, lineType=cv2.LINE_AA)
        if nb_ch == 4:
            # image, buffer have depth 4
            # Alpha blend buffer
            # buffer contains RGBA data, with per pixel alpha
            alpha = buffer.channel(3)
        else:
            # Draw alpha mask
            alpha = np.zeros_like(buffer.channel(0), dtype=np.uint8, subok=True)
            cv2.fillConvexPoly(alpha, pts, color=[alphaval], shift=shift, lineType=cv2.LINE_AA)
        alpha.binary = False
        alpha = alpha.to_float()

        if not(image.warped):
            buffer = unwarp(buffer)
            alpha = unwarp(alpha)

        if image.dtype == np.float32 or image.dtype == np.float64:
            image *= (1-alpha)
            image += (alpha*buffer)
        else:
            # Integers
            image[:] = (image*(1-alpha)).astype(np.uint8)
            image +=   (alpha * buffer).astype(np.uint8)
        return None
