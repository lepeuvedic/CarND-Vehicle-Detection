import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
import warnings
from weakref import WeakValueDictionary as WeakDict
from .decorators import strict_accepts, generic_search, flatten_collection
from .decorators import _make_hashable, static_vars
from .CameraCalibration import CameraCalibration
from .Line import Line
from .LinePoly import LinePoly
import itertools
from functools import partial, partialmethod
from threading import current_thread
from random import shuffle
from collections import deque

class _Record(object):
    pass

class RoadImage(np.ndarray):

    unique_id = 0
    
    # * syntax makes all the following arguments keyword only: they must be used with keyword=value syntax
    # Here they have default values, so they can be missing as well.
    @strict_accepts(object, (None, np.ndarray))
    def __new__(cls, input_array=None, *, filename=None, cspace=None, src_cspace=None):
        """
        Create a new roadimage, or load new data in an existing one.
        If input_array is given, it can be an existing RoadImage or any numpy array.
        The returned object will share memory with input_array (they will use the same underlying buffer).
        If input_array is not given, a filename must be given.
        If a filename is given and the corresponding file can be read, the image will be resized and converted
        into the shape and dtype of input_array if input_array is given, and returned in a new buffer at its
        normal size otherwise.
        The data is converted from src_cspace to cspace colorspace. src_cspace is the colorspace of the 
        file. If filename is None, it is the colorspace of input_array instead.  
        """
        # Default parameter values
        # src_cspace
        # If filename is given, src_cspace overrides the assumption that mpimg.imread will return RGB. We assume that
        # the caller knows what is stored in the file. If it is not given and we read from a file, the assumption holds.
        if src_cspace is None:
            if filename:
                # Will be set below when  mpimg.imread has read the file.
                src_cspace = 'data'
            else:
                if issubclass(type(input_array), cls):
                    # Get from input RoadImage
                    src_cspace = input_array.colorspace
                else:
                    # Cannot guess...
                    raise ValueError('RoadImage: Cannot guess color encoding in source. Use src_cspace argument.')
            
        # cspace
        if cspace is None:
            if filename and not(input_array is None):
                raise ValueError('RoadImage: Cannot use input_array and filename together.')
            # Unless a specific conversion was requested with cspace=, do not convert source color representation.
            cspace = src_cspace

        # img is the src_cspace encoded data read from a file.
        img = None
        if filename:
            # Read RGB values as float32 in range [0,1]
            img = mpimg.imread(filename)
        else:
            img = input_array
            
        if img is None:
            raise ValueError('RoadImage: Either input_array or filename must be passed to constructor.')
        
        # Invariant: img is an instance of np.ndarray or a derivative class (such as RoadImage)
        
        if RoadImage.is_grayscale(img):
            # No immediate conversion in ctor for grayscale images
            # Correct defaults
            if src_cspace == 'data': src_cspace = 'GRAY'
            if cspace == 'data':     cspace = 'GRAY'
            # Normalize shape
            if img.shape[-1] != 1:
                img = np.expand_dims(img,-1)
        else:
            if src_cspace == 'data': src_cspace = 'RGB'
            if cspace == 'data':     cspace = 'RGB'
            
        if src_cspace != cspace and (src_cspace != 'RGB' or cspace == 'GRAY'):
            # Not 'RGB' nor cspace: fail because we won't do two convert_color 
            raise ValueError('RoadImage: Cannot autoconvert from '+str(src_cspace)+' to '+str(cspace)+' in ctor.')

        # Invariant: (src_cspace == 'RGB' and cspace != 'GRAY') or src_cspace == cspace

        # Change colorspace (from 3 channel to 3 channel only)
        if cspace != src_cspace:
            # Invariant: src_cspace == 'RGB' and cspace != 'GRAY'
            cv2_nbch = RoadImage.cspace_to_nb_channels(cspace)
            assert cv2_nbch == 3, 'RoadImage: BUG: Check number of channels for cspace '+cspace+" in CSPACES"
            cv2_code = RoadImage.cspace_to_cv2(cspace)  # Returns None for cspace='RGB' since we are already in RGB.
            if cv2_code:
                cv2.cvtColor(img, cv2_code, img)  # in place color conversion

        # Create instance and call __array_finalize__ with obj=img
        # __array_finalize__ declares the new attributes of the class
        obj = img.view(cls)
        
        # Set colorspace (for new instances, __array_finalize__ gets a default value of 'RGB')
        obj.colorspace = cspace
 
        # Set filename
        if filename: obj.filename = filename
        # Finally, we must return the newly created object:
        return obj

    def _inherit_attributes(self, obj, binary_test=True):
        """
        Inherit the values of attributes from obj.
        If obj is a road image, those attributes are copied.
        Line instances are shared in a family of related images.
        If obj is a numpy array, those attributes are initialized to default values.
        """
        # By default binary is False, but for __new__ RoadImages (obj is None or numpy array),
        # an attempt is made to assess binarity.
        try:
            self.binary = True
            self.binary = obj.binary  # may raise AttributeError
        except AttributeError:
            pass
        if self.binary and binary_test:
            # Lots of images made only from binary images are not binary.
            # For instance, the np.sum() of a collection of binary images has integer pixel values,
            # but is not always binary. Normal values of binary images are -1 (signed only), 0, 1.
            # Perform expensive test:
            bcast = np.broadcast(1,self)
            v = np.broadcast_to(self, bcast.shape)  # returns a numpy.ndarray
            maxi = v.max()
            mini = v.min()
            if maxi.dtype == np.float32: epsilon = 10. * np.finfo(type(maxi)).eps
            else: epsilon = 1
            if not(np.isfinite(maxi) and np.isfinite(mini)) or\
               not(mini == maxi or mini == 0 or abs(mini+maxi)<epsilon):
                # It implies that the max or the min is not in the set { -1, 0 , 1 }.
                self.binary = False
            else:
                ones  = np.broadcast_to(maxi, bcast.shape)
                zeros = np.broadcast_to(0, bcast.shape)
                mones  = np.broadcast_to(mini, bcast.shape)  # for minus one

                if np.int8(mini) >= 0:
                    # Use binary test for non-gradient (unsigned) images
                    mask = ((v==ones) | (v==zeros))
                else:
                    mask = ((v==ones) | (v==zeros) | (v==mones))
                self.binary = bool(mask.all())
        else:
            # Non binary ancestor or no binary_test
            self.binary = False

        #self.serial = getattr(obj, 'serial', 0) + 1           # Helps with code tracing (DEBUG)
        self.warped = getattr(obj, 'warped', False)           # True for images returned by 'warp'
        self.colorspace = getattr(obj, 'colorspace', 'RGB')   # inherited, set by __new__ for new instances
        self.gradient = getattr(obj, 'gradient', False)       # True for a gradient image: inherited
        self.undistorted = getattr(obj, 'undistorted', False) # True for undistorted images
        self.filename = getattr(obj, 'filename', None)        # filename is inherited

        # Camera calibration is defined by the call to 'undistort'.
        # A few methods only make sense on camera images: warp and unwarp are examples.
        self.calibration = getattr(obj, 'calibration', None)  # CameraCalibration instance
        
        # Share the state of the find_lines method. This method is a high level method which
        # calls lower level methods, and sharing this allows different methods to access the
        # state data on any related road image.
        self.find_lines_state = getattr(obj, 'find_lines_state', None)
        if self.find_lines_state is None: del self.find_lines_state
        
        # Share the state of the find_cars method. Like find_lines, find_cars accumulates information
        # from time series and maintains state information. Copying the attribute here only helps
        # accessing the state data from any related road image.
        self.find_cars_state = getattr(obj, 'find_cars_state', None)
        if self.find_cars_state is None: del self.find_cars_state
        
        return
        
    def __array_finalize__(self, obj):
        # When called from __new__,
        # ``self`` is a new object resulting from ndarray.__new__(RoadImage, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # In other cases, it is called directly from the constructor of ndarray, when calling .view(RoadImage).
        #
        # Add attributes with default values, or inherited values
        
        # Cropping:
        # A cropped area is an ndarray slice. They shares the same data, therefore a slice can be used
        # to modifiy the original data.
        # crop_area is computed when a slice is made, and otherwise is None
        # not change the width or depth. When a cropping is cropped again, a chain is created.

        # A method get_crop(self,parent) computes the crop area relative to the given parent.
        # A method crop_parents(self) iterates along the chain.
        
        self.crop_area   = None           # A tuple of coordinates ((x1,y1),(x2,y2))
        
        # The parent image from which this image was made.
        # Always initialized to None, it is set by methods in this class which return a new RoadImage instance.
        # A method parents(self) iterates along the chain.
        self.parent = None
        
        # A dictionary holding child images, with the generating operation as key. No child for new images.
        # Children from a tree. Branches which are no longer referenced may be deleted.
        self.child = WeakDict()
        
        # By default inherited
        self._inherit_attributes(obj)
        
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. RoadImage():
        #    obj is None
        #    (we're in the middle of the RoadImage.__new__ constructor)
        if obj is None:
            # We never call super() to initialize new instance. All instances are created
            # with .view(RoadImage) from a numpy array.
            assert not(obj is None), 'RoadImage.__array_initialize__: BUG: __init__ from None!?'

        # From view casting - e.g img.view(RoadImage):
        #    obj is img
        #    (type(obj) can be RoadImage, but can also be numpy array)
        #    Typically: np.copy(roadimage).view(RoadImage)
        if issubclass(type(obj), np.ndarray):
            # Compute self.crop and op
            bounds = np.byte_bounds(obj)
            crop_bounds = np.byte_bounds(self)
            is_inside = (crop_bounds[0]>=bounds[0]) and (crop_bounds[1]<=bounds[1])
            if is_inside:
                # Compute crop_area x1,y1
                #print('Compute crop: self='+str(crop_bounds)+'  parent='+str(bounds))
                #print('Shape:   self='+str(self.shape)  +'  parent='+str(obj.shape))
                #print('Strides: self='+str(self.strides)+'  parent='+str(obj.strides))
                # First corner
                byte_offset = crop_bounds[0] - bounds[0] 
                assert byte_offset <= bounds[1]-bounds[0], \
                        'RoadImage:__array_finalize__ BUG: Error in crop_area 1 computation.'
                # can be equal if both self and obj have collapsed to zero size in one dimension.
                # In this case, bounds[0] effectively points *after* the end (and equals bounds[1]).

                # Find coords
                coords1 = []
                for n in obj.strides:
                    if n == 0:
                        w = 0
                    else:
                        w = byte_offset//n
                        byte_offset -= n*w
                    coords1.append(w)
                assert byte_offset == 0, 'RoadImage:__array_finalize__ BUG: crop_area 1 computation: item_offset != 0.'

                # Second corner (crop_bounds[1] is 1 item after the last element of self)
                byte_offset = crop_bounds[1] - bounds[0]
                if byte_offset == 0:
                    raise ValueError('RoadImage.crop: Cropped area has size zero.')
                byte_offset -= self.itemsize
                assert byte_offset <= bounds[1]-bounds[0], \
                        'RoadImage:__array_finalize__ BUG: Error in crop_area 2 computation.'

                # Find coords
                coords2 = []
                for n in obj.strides:
                    if n == 0:
                        w = 0
                    else:
                        w = byte_offset//n
                        byte_offset -= n*w
                    coords2.append(w+1)
                assert byte_offset == 0, 'RoadImage:__array_finalize__ BUG: crop_area 2 computation: item_offset != 0.'
                crop_area = (tuple(coords1),tuple(coords2))
        
                # We have the n-dimensional crop_area... and we store the last three dimensions.
                self.crop_area = (tuple(coords1[-3:]),tuple(coords2[-3:]))
                
                if coords1 == [0]*len(coords1) and coords2 == list(obj.shape):
                    # For a reshape, coords1 == [0,0,...] and coords2 == list(obj.shape).
                    # It is never a crop, but we can also have eliminated the G channel in an RGB image.
                    # Because we handle images, we must ensure that the last three dimensions are the same
                    if self.shape[-3:] != obj.shape[-3:]:
                        if self.ndim == 1:
                            # Allow ravel()
                            op = ((np.ravel,),)
                        elif self.shape[-3:-1] == obj.shape[-3:-1] and self.strides[-3:-1] == obj.strides[-3:-1]:
                            # It's a channels operation which kept the first and last channels and eliminated at last one
                            # The step can be deduced from the number of channels in obj and self. The // is exact.
                            op = ((RoadImage.channels, (range(0,obj.shape[-1],(obj.shape[-1]-1)//(self.shape[-1]-1)),)),)
                        else:
                            raise NotImplementedError('RoadImage.__array_finalize__ BUG: '
                                                      +'Bad reshape operation done by caller' )
                    else:
                        op = ((RoadImage.crop, (self.crop_area,)),)
                elif self.strides[-3:-1] == obj.strides[-3:-1]:
                    # Some dimensions may be gone but only in the collection layout, and the block is dense,
                    # meaning that the slice arguments did not use a step different from 1.
                    # The width and height strides are the same: it's a crop, maybe with a channels()
                    if coords1[-1] == 0 and coords2[-1] == obj.shape[-1]:
                        op = ((RoadImage.crop, (self.crop_area,)),)
                    elif self.strides[-1] == obj.strides[-1] or self.shape[-1]==1:
                        # Extraction of contiguous channels, with or without a crop, and maybe just 1 channel.
                        op = ((RoadImage.crop, (self.crop_area,)),(RoadImage.channels,(coords1[-1],coords2[-1])))
                    else:
                        raise NotImplementedError('RoadImage.__array_finalize__ BUG: '
                                                  +'Bad reshape operation done by caller' )
                elif np.prod(np.array(coords2)-np.array(coords1)) == self.size:
                    # Dense selection: the crop information captures it all.
                    if coords1[-1] == 0 and coords2[-1] == obj.shape[-1]:
                        op = ((RoadImage.crop, (self.crop_area,)),)
                    else:
                        op = ((RoadImage.crop, (self.crop_area,)),(RoadImage.channels,(coords1[-1],coords2[-1])))
                else:
                    # General slice: may be keeping 1 pixel in 2
                    # Slicing operations cannot be automatically replayed
                    RoadImage.unique_id += 1
                    op = ((RoadImage.__slice__, (RoadImage.unique_id,)),)
                    warnings.warn('RoadImage.__array_finalize__: deprecated slicing.', DeprecationWarning)
                    # Those cases must be corrected in the caller
            
            # From new-from-template - e.g img[:3]
            #    type(obj) is RoadImage
            # If the object we build from is a RoadImage, we link child to parent and parent to child
            # but @generic_search will overwrite this link for operations other than those inferred above.
            if issubclass(type(obj), RoadImage):
                # If is_inside, op is already set.
                if not(is_inside):
                    if obj.shape == self.shape:
                        # Operation is generic, but registers what make self unique.
                        RoadImage.unique_id += 1
                        op=((RoadImage.__numpy_like__, (RoadImage.unique_id,)),)
                    else:
                        op=None
                        
                if op:  obj.__add_child__(self, op, unlocked=True)
                    
                
            return
        raise TypeError("RoadImage: cast of %s to RoadImage is invalid."% str(type(obj)))
        # We do not need to return anything

    def __del__(self):
        """
        Instances are linked by self.parent (a strong reference) and self.child dictionary (weak references).
        Children which are not directly referenced may be deleted by the garbage collector.
        """
        if self.parent is None:
            return
        # Check if parent still has children and make writeable again if not.
        if RoadImage.__has_only_autoupdating_children__(self.parent, excluding = self) :
            if not(self.parent.flags.writeable):
                #print('unlocking parent')
                try:
                    self.parent.flags.writeable = True
                except Exception as e:
                    print('RoadImage.__del__:%s' % str(e))
                    print('op =',str(self.find_op()),' parent op =',str(self.parent.find_op()))
        # Stop referencing parent
        self.parent = None
        
    def unlink(self):
        """
        Detaches a RoadImage from its parent.
        Throws an exception if self shares data with his parent (slice, crop, reshape, ...).
        TODO: If data was shared with parent and self's descendents who also shared data
              with the parent now share data with self. 
        """
        # When data is shared, self.crop_area is not None
        if self.shares_data(self.parent):
            raise ValueError('RoadImage.unlink: Cannot unlink views that share data with parent.')
        # Remove self from parent's children
        if not(self.parent is None):
            op = self.find_op(raw=True)
            if op: del self.parent.child[op]
            #for op, sibling in self.parent.child.items():
            #    if sibling is self:
            #        del self.parent.child[op]
            #        break # Do not continue iterating on modified dictionary
            # If parent no longer has children, make writeable again
            if RoadImage.__has_only_autoupdating_children__(self.parent) : self.parent.flags.writeable = True
        self.parent = None
        
        
    CSPACES = {
        'RGB': (None, None, 3), 
        'HSV': (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB, 3), 
        'HLS': (cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB, 3),
        'YUV': (cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 3),
        'LUV': (cv2.COLOR_RGB2LUV, cv2.COLOR_LUV2RGB, 3),
        'XYZ': (cv2.COLOR_RGB2XYZ, cv2.COLOR_XYZ2RGB, 3),
        'YCC': (cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 3),
        'LAB': (cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB, 3),
        'GRAY':(cv2.COLOR_RGB2GRAY,cv2.COLOR_GRAY2RGB, 1)
    }

    SOBELMAX = { 3: 4, 5: 48, 7: 640, 9: 8960 }
    
    # Ancillary functions (not methods!)    
    @classmethod
    def __has_only_autoupdating_children__(cls, obj, *, excluding=None):
        """
        Returns True if obj children are all views sharing self data.
        """
        for ops, ch in obj.child.items():
            for op in ops:
                # Check each op in long ops
                # With decorators, the function pointer stored in the tuple is not the same function
                if not(op[0].__name__ in cls.AUTOUPDATE):
                    # Found an op which does not support automatic updates
                    if not(ch is excluding): return False
        return True
    
    @classmethod
    def cspace_to_cv2(cls, cspace):
        if not(cspace in cls.CSPACES.keys()):
            raise ValueError('cspace_to_cv2: Unsupported color space %s' % str(cspace))
        return cls.CSPACES[cspace][0]
        
    @classmethod
    def cspace_to_cv2_inv(cls, cspace):
        if not(cspace in cls.CSPACES.keys()):
            raise ValueError('cspace_to_cv2: Unsupported color space %s' % str(cspace))
        return cls.CSPACES[cspace][1]

    @classmethod
    def cspace_to_nb_channels(cls, cspace):
        if not(cspace in cls.CSPACES.keys()):
            raise ValueError('cspace_to_cv2: Unsupported color space %s' % str(cspace))
        return cls.CSPACES[cspace][2]

    @classmethod
    def image_channels(cls, img):
        """
        Returns the number of channels in an image. Whereas width and depth are usually, but not always, large, the number
        of channels is usually, but not always 1 or 3.
        The function assumes a table of images if img is 4D.
        If the last dimension is length 1 or 3, a single pixel (color value), a vector of pixels or an image is assumed.
        """
        if not(issubclass(type(img), np.ndarray)):
            raise TypeError('image_channels: img must be a numpy array or an instance of a derivative class.')
        if issubclass(type(img), RoadImage):
            # It is always a single image
            if RoadImage.is_grayscale(img): return 1
            return img.shape[-1]
        size = img.shape
        if len(size)==4: return size[-1]
        # Cannot say for 3x3 kernel
        if size == (3,3):
            raise ValueError('image_channels: 3-by-3 numpy array can be either a small kernel or a vector '
                             +'of 3 color pixels. Store kernel as (3,3,1) and vector as RoadImage to remove ambiguity.')
        if size[-1] == 1 or size[-1] == 3: return size[-1]
        if len(size) == 1 or len(size) == 2: return 1
        raise ValueError('image_channels: Cannot guess which dimension, if any, is the number of channels in shape %s.' 
                         % str(size))
                                       
    @classmethod
    def __match_shape__(cls, shape, ref):
        """
        Adds singleton dimensions to shape to make it generalize to the reference shape ret.
        """
        assert issubclass(type(shape),tuple), 'RoadImage.__match_shape__: BUG: shape must be a tuple.'
        assert issubclass(type(ref),tuple), 'RoadImage.__match_shape__: BUG: ref must be a tuple.'
        out = [1]*len(ref)
        if shape == (1,):
            # Scalar case
            pass
        elif shape == (ref[-1],):
            # Per channel thresholds
            out[-1] = ref[-1]
        elif shape == (ref[-3], ref[-2]) or shape == (1,) + (ref[-3], ref[-2]):
            # Per pixel thresholds, same for all channels
            out[-3] = ref[-3]
            out[-2] = ref[-2]
        elif shape == ref[-3:] or shape == (1,) + ref[-3:]:
            # Per pixel thresholds, different for each channel
            out[-3] = ref[-3]
            out[-2] = ref[-2]
            out[-1] = ref[-1]
        elif shape == ref:
            # Per pixel, per channel and per image thresholds for the whole collection
            # Can be used to compare collections.
            out = list(ref)
        else:
            # Anything that generalizes to shape
            assert len(shape) == len(ref), \
                'RoadImage.threshold: min must be a documented form or have the same number of dimensions as self.'
            out = list(shape)
            ref_list = list(ref)
            assert all( m==1 or m==s for m,s in zip(out, ref_list)), \
                'RoadImage.threshold: Invalid array shape.'
        return tuple(out)
            
    @classmethod
    def is_grayscale(cls, img):
        """
        Tells if an image is encoded as grayscale
        """
        if issubclass(type(img), RoadImage):
            # the function is called by image_channels if img is a RoadImage: must decide without relying on it.
            return img.colorspace == 'GRAY'
            
        # In other cases it depends on the shape and number of channels
        if not( issubclass(type(img), np.ndarray) ):
            raise ValueError('is_grayscale: img must be a numpy array or an instance of a derivative class.')
        size = img.shape

        if size[-1] == 3: return False  # It can be color specification if len(size)==1, or a vector of color pixels.
        return len(size)==1 or len(size)==2 or (len(size)==3 and size[-1]==1)

    @classmethod
    def __find_common_ancestor__(cls, lst):
        """
        Returns the shortest possible list of ancestors, and a list of indices into that list, which corresponds 
        to lst element-by-element.
        Internal function: assumes lst is not empty and lst elements are all RoadImage instances.
        """
        ancestry = []
        for img in lst:
            lineage = [img]
            for p in img.parents():
                lineage.insert(0,p)
            ancestry.append(lineage)
        out_index = [None]*len(lst) # a different list
        
        # We have gone as far back as possible in ancestors. All the elements which have an ancestor in common
        # are associated to lists beginnng with the same ancestor. The number of unique very old ancestors is
        # the final number of ancestors.
        
        # extract unique ancestors, not necessarily the youngest ones
        ancestors = []
        for i,lineage in enumerate(ancestry):
            # NB: "if lineage[0] in ancestors" fails since in seems to get distributed over the numpy array
            if any(lineage[0] is anc for anc in ancestors):
                out_index[i] = ancestors.index(lineage[0])
            else:
                out_index[i] = len(ancestors)
                ancestors.append(lineage[0])

        out_ancestors = []
        # We now have len(ancestors) independant families
        # Within each family, the goal is to find the youngest ancestor
        for family, anc in enumerate(ancestors):
            # Gather related ancestries
            family_ancestry = [ lineage for i,lineage in enumerate(ancestry) if out_index[i]==family ]
            ref_anc = family_ancestry[0]
            # Single element family
            if len(family_ancestry)==1:
                # Return lst element itself
                out_ancestors.append(ref_anc[-1])
            else:
                # Descend ancestries as long as ancestors are identical
                i = 0
                equal = True
                last = min(len(ref_anc),len(lineage))  # Minimum value is 1.
                while equal:
                    i += 1
                    if i==last: break  # ref_anc and lineage are direct line ancestors
                    for lineage in family_ancestry[1:]:
                        equal = (equal and (lineage[i] is ref_anc[i])) 
                out_ancestors.append(ref_anc[i-1])

        return out_ancestors, out_index
    
    @classmethod
    def find_common_ancestor(cls, lst):
        """
        Finds a common ancestor to a list of RoadImage instance, using lst[n].parent.
        Returns a minimum list of ancestors. The returned ancestors could be the elements of lst themselves
        if they do not share an ancestor with other elements of lst.
        The function also returns a list of indices, the same length as lst, which associates each element of lst
        with one ancestor.
        """
        if len(lst)==0:
            raise ValueError('RoadImage.find_common_ancestor: lst must not be an empty list.')
        for i,img in enumerate(lst):
            if not(issubclass(type(img),RoadImage)):
                raise TypeError('RoadImage.find_common_ancestor: List elements must be type RoadImage. Check element %d.'
                                % i)
        return RoadImage.__find_common_ancestor__(lst)
    
    @classmethod
    def select_ops(cls,ops, selected):
        """
        Processes a sequence of operations to keep only those selected.
        selected must be a list of RoadImage method names ('convert_color', 'to_grayscale', ...).
        ops must be a long op, as returned by find_op(raw=False).
        """
        if ops == ():
            return ()
        assert issubclass(type(ops[0]), tuple), 'RoadImage.select_ops: BUG ops format is invalid: '+repr(ops)
        ret = ()
        for op in ops:
            if op[0] in selected: ret += (op,)
        return ret
    
    @classmethod
    def pretty_print_ops(cls, ops, trim=200):
        """
        Takes an ops key stored in the child attribute of any image, and returns a nice string representation.
        """
        def get_size(t, trim):
            """
            Returns the total size of t, unless it is larger than trim, in which case a value
            larger than trim, but probably much smaller than the real size of t, is returned.
            """
            from sys import getsizeof
            # getsizeof does not give the total size of dicts, tuples or lists
            sz = 0
            if issubclass(type(t),dict):
                for key,val in t.items():
                    sz += get_size(val, trim)
                    if sz > trim: return sz
                    sz += get_size(key, trim)
                    if sz > trim: return sz

            elif issubclass(type(t),(tuple,list)):
                for e in t:
                    sz += get_size(e, trim)
                    if sz > trim: return sz

            else:
                sz = getsizeof(t)
            return sz

        def to_string(a):
            if get_size(a, trim) > trim:
                return '...hash(' + str(hash(a)) + ')...'
            return repr(a)
        
        # Using a simple tuple for simple ops is deprecated. Even single ops shall be stored as tuple of tuple.
        if ops:
            assert type(ops[0]) is tuple, 'RoadImage.pretty_print_ops: BUG operation format is invalid: '+str(ops[0])
            out = []
            for op in ops:
                # Process each op: there are at most three
                # (f, args, kwargs), where args and kwargs are optional and kwargs is a dict
                if not(isinstance(op[0],str)):
                    # It is either a function handle or a str
                    pretty_op=[ op[0].__name__ ]
                else:
                    pretty_op=[ op[0] ]
                if pretty_op[0] == RoadImage.make_collection.__name__:
                    # Special case because arguments are ops
                    for a in op[1:]:
                        pretty_op.append(RoadImage.pretty_print_ops(a, trim=trim))
                else:
                    for a in op[1:]:
                        if not(isinstance(a, tuple) and len(a)>0):
                            raise ValueError('RoadImage.pretty_print_ops: Invalid operation format: '+str(a))
                        if a[0] is dict:
                            # named arguments kwargs
                            for param in a[1:]:
                                if not(len(param)==2):
                                    raise ValueError('RoadImage.pretty_print_ops: Invalid named parameter format.')
                                pretty_op.append(str(param[0])+'='+to_string(param[1]))
                        else:
                            # positional arguments (will always be first)
                            for param in a:
                                pretty_op.append(to_string(param))
                pretty_op = '( ' + ", ".join(pretty_op) + ' )'
                out.append(pretty_op)
            out = '(' + ", ".join(out) + ')'
            return out
        return '()'
            
    @classmethod
    def make_collection(cls, lst, size=None, dtype=None, concat=False):
        """
        Takes a list of RoadImage objects and returns a collection.
        It also accepts a list of collections, but they must have the same structure. In this case, a one dimension
        higher collection is returned, unless concat is True, in which case make_collection accepts collections
        having axis 0 of different lengths (they are concatenated along axis 0).
        Images are resized/uncropped to fit the given size, or the shape of a common parent if any.
        If channels were extracted, and some elements of lst are multi-channel, the result will be multi-channel
        with single channel elements placed at the right position and padded with zeroed channels.
        If dtype is given and is compatible (uint8 incompatible with negative data), the returned collection
        is cast and scaled for that dtype.
        make_collection is important to prepare batches for tensorflow or Keras.
        NB: This implementation only validates that all collections in lst have the same dtype.
        NB: The resulting collection has no parent.
        """
        if len(lst)==0:
            raise ValueError('RoadImage.make_collection: List lst must not be an empty list.')
        for i,img in enumerate(lst):
            if not( issubclass(type(img),RoadImage) ):
                raise ValueError('RoadImage.make_collection: List elements must be type RoadImage. Check element %d.'% i)
        if dtype is None:
            dtype = lst[0].dtype
        assert len(lst[0].shape) >= 3, 'RoadImage.make_collection: BUG: Found RoadImage with less then 3 dims in lst.'
        shape_coll = lst[0].shape[:-3]
        if concat and len(shape_coll)==0:
            raise ValueError('RoadImage.make_collection: elements must be collections to use concat=True.')
        for i, img in enumerate(lst):
            if img.shape[:-3] != shape_coll:
                if not(concat) or img.shape[1:-3] != shape_coll[1:]:
                    raise ValueError('RoadImage.make_collection: List elements must have the same collection structure. '
                                     + 'Check element %d.' % i)
        # Future implementation will cast dtype intelligently. Current one requires same dtype.
        # Future implementation will handle 'channel' op. Current one requires same number of channels
        nb_ch = RoadImage.image_channels(lst[0])
        for i,img in enumerate(lst):
            if img.dtype != dtype:
                raise ValueError('RoadImage.make_collection: List elements must share the same dtype. '+
                                 'Check element %d.' % i)
            if RoadImage.image_channels(img) != nb_ch:
                raise ValueError('RoadImage.make_collection: List elements must have the same number of channels. '+
                                 'Check element %d.' % i)

        ancestors, index = RoadImage.__find_common_ancestor__(lst)
        # Future implementation will manage several ancestors
        if len(ancestors) > 1:
            # Make collection and copy cut the link to the ancestor, so it may not be abnormal
            # TODO: link the collection to the common ancestor, with as parameters, all the required operations.
            warnings.warn('RoadImage.make_collection: cannot be sure that all the elements have the same ancestor.',
                          RuntimeWarning)
            # All ancestors must have the same size
            for anc in ancestors[1:]:
                if anc.get_size() != ancestors[0].get_size():
                    raise ValueError('RoadImage.make_collection: List elements must have the same ancestor.')

        crop_area = lst[0].get_crop(ancestors[index[0]])
        ref_ops = lst[0].find_op(ancestors[index[0]], raw=False)
        # Keep sequence of crops and warps
        KEY_OPS = ['crop', 'warp', 'resize']
        ref_ops = RoadImage.select_ops(ref_ops, KEY_OPS)
        
        for i,img in enumerate(lst):
            # We only need to cater about resize, warp and crop ops
            # All images must have the same crop versus ancestor
            msg = 'RoadImage.make_collection: List elements must show the same crop area {0}. Check element {1}'
            if img.get_crop(ancestors[index[i]]) != crop_area:
                raise ValueError(msg.format(str(crop_area) , str(i)))
            # All images must have identical warp operations in the same relative order with crop operations
            # Find sequence of operations from common ancestor to list element
            ops = RoadImage.select_ops(img.find_op(ancestors[index[i]], raw=False), KEY_OPS)
            if ops != ref_ops:
                raise ValueError('RoadImage.make_collection: '+
                                 'List elements must have the same sequence of crops, warps and resizes.')
            
        # All is ok
        # Stack all the elements of lst. stack make a new copy in memory.
        if concat:
            coll = np.concatenate(lst, axis=0).view(RoadImage)
        else:
            coll = np.stack(lst, axis=0).view(RoadImage)
        coll.gradient = all([img.gradient for img in lst])
        coll.binary = all([img.binary for img in lst])
        coll.colorspace = lst[0].colorspace
        if any([img.colorspace != coll.colorspace for img in lst]): coll.colorspace = None
        coll.crop_area = lst[0].crop_area
        if any([img.crop_area != coll.crop_area for img in lst]): coll.crop_area = None
        coll.filename = lst[0].filename
        if any([img.filename != coll.filename for img in lst]): coll.filename = None
        # No parent, because make_collection is not an operation on a single image
        coll.parent = None
        # If we have a single ancestor, add the collection as descendant
        # TODO: try to reuse existing recorded collections...
        if len(ancestors)==1:
            oplst = [ img.find_op(ancestors[index[i]], raw=True) for i, img in enumerate(lst) ]
            op = ( tuple([RoadImage.make_collection] + oplst + [ ( dict , ('concat', concat)) ] ), )
            ancestors[0].__add_child__(coll, op)
        else:
            warnings.warn('RoadImage.make_collection: Cannot link to parent when there are %d ancestors.'
                          % len(ancestors), RuntimeWarning)
        return coll
    
    def get_size(self):
        if len(self.shape) >= 3:
            # Last dimension is channels and will be 1 for grayscale images
            return (self.shape[-2], self.shape[-3])
        elif len(self.shape) == 2:
            # Is a vector
            return (self.shape[-2],1)
        else:
            # Is a color
            return (1,1)
    
    def parents(self):
        """
        Generator function going up the list of parents.
        """
        p = self.parent
        while not(p is None):
            yield p
            assert issubclass(type(p), RoadImage) , 'RoadImage.parents: BUG: parent should be RoadImage too.'
            p = p.parent
        
    def crop_parents(self):
        """
        Generator function going up the list of crop-parents.
        """
        p = self
        while not(p.parent is None):
            assert issubclass(type(p.parent), RoadImage) , \
                'RoadImage.crop_parents: BUG: crop parent should be RoadImage too.'
            if not(p.crop_area is None):
                yield p.parent
            p = p.parent
    
    def get_crop(self, parent):
        """
        In case of crop of crop, the crop_area variable only contains the crop relative to the immediate parent.
        This utility method computes the crop area relative to any parent.
        """
        if self.ndim <3:
            raise ValueError('RoadImage.get_crop: image must have shape (height,width,channels).')
        p = self
        x1 = 0
        y1 = 0
        x2 = self.shape[-2]
        y2 = self.shape[-3]
        while not(p is None) and not(p is parent):
            # Add x1,y1 of parent to x1,y1 and x2,y2 of self
            if not(p.crop_area) is None:
                xyc = p.crop_area[0]
            else:
                xyc = (0,0,0)
            p = p.parent
            if not(p is None):
                y,x = xyc[:2]
                x1 += x
                x2 += x
                y1 += y
                y2 += y
        return ((x1,y1),(x2,y2))
    
    # Functions below all manipulate the 'child' dictionary. The dictionary associates an op (a method used to derive
    # an image represented as a tuple) to another RoadImage instance, which is the result of that op.
    # Only the parent may have links to a RoadImage in his parent.child dictionary.
    def children(self):
        """
        Generator function going across the dictionnary of children, returning handles to existing RoadImage instances.
        """
        for k in self.child.keys():
            yield self.child[k]
            
    def list_children(self):
        """
        Lists available children as a tree of tuples. Recursive.
        """
        l = []
        for k in self.child.keys():
            ch = self.child[k].list_children()
            if ch:
                l.append([RoadImage.pretty_print_ops(k) , ch])
            else:
                l.append(RoadImage.pretty_print_ops(k))
        return l
    
    def find_child(self, op):
        """
        Returns the RoadImage object associated with operation op.
        The method may be specified as a string, same as output by find_op.
        In most cases, find_child does just an access to the self.child dictionary.
        """
        # Due to the use of decorators, RoadImage.xxx is not the actual method xxx
        # recorded in child.keys. The actual recorded function has the same name,
        # but is a particular instantiation of one of the decorators.
        # TODO : Should look at grand-children in case of long op
        # NOTDONE : cache optimizations never send long ops
        is_long_op = (issubclass(type(op[0]),tuple) and len(op)>1)
        if is_long_op:
            raise NotImplementedError('RoadImage.find_child: long operations are not yet implemented.')
        # Fast method for ops obtained using find_op(raw=True)
        try:
            return self.child[op]
        except KeyError:
            pass
        # Slower method based on __name__ comparison, only if op[0][0] is a string
        strop = op[0][0]
        if issubclass(type(strop),str):
            for kop in self.child.keys():
                # The first implementation stored ops as tuple(method, args). The current one wraps several ops in
                # another tuple, using the same format for long and for short ops.
                assert issubclass(type(kop[0]),tuple), 'RoadImage.find_child: BUG: child.keys() still contains short ops!'
                if len(kop)>1:
                    # Search for long ops is not implemented: skip long ops.
                    continue
                if kop[0][0].__name__ == op[0][0]:
                    # If method name matches, make a raw op using method handle and try fast method
                    raw_op = ((kop[0][0],)+ op[0][1:],)
                    try:
                        return self.child[raw_op]
                    except KeyError:
                        # If the key isn't in self.child, it isn't there: stop iterations
                        break
        return None
    
    def shares_data(self, parent=None):
        """
        Returns true if self is view on parent's data.
        parent defaults to the immediate parent.
        """
        if parent is None:
            parent = self.parent
        if parent is None:
            # Has no parent --> shares nothing
            return False
        bounds = np.byte_bounds(parent)
        crop_bounds = np.byte_bounds(self)
        is_inside = (crop_bounds[0]>=bounds[0]) and (crop_bounds[1]<=bounds[1])
        return is_inside
        
    def find_op(self, parent=None, *, raw=False, quiet=False):
        """
        Returns a tuple of tuples which describes the operations needed to make self
        from its parent. Note that if self has been modified in-place, there will be more than one operation.
        The function returns whatever has been recorded by operations. Only RoadImage operations properly
        record what they do.
        If quiet is True, the function will return None instead of raising an exception if parent does
        not have an operation associated with self, or if parent is not among self's ancestors.
        raw mode has find_op return actual keys from the dictionaries holding the chilren, rather than human-readable.
        """
        if parent is self:
            return ()
        if self.parent is None:
            # self is top of chain
            if not(parent is None) and not(quiet):
                # If the caller gave an explicit ancestor in parent, he got it wrong.
                raise ValueError('RoadImage.find_op: parent not found among ancestors.')
            return ()
        if parent is None:
            # Default parent for search is immediate one
            parent = self.parent
        # Higher up search (recursive)
        if not(parent is self.parent):
            ops = self.parent.find_op(parent, raw=raw)
            if ops:
                assert issubclass(type(ops[0]),tuple) , \
                    'RoadImage.find_op: BUG: find_op did not return tuple of tuple.'
        else:
            ops = ()

        def substitute_name(longop):
            ret = []
            for op in longop:
                name = op[0].__name__
                if name == RoadImage.make_collection.__name__:
                    # Other args are ops too, except the last which encodes concat=T/F
                    ret.append((name,)+tuple([substitute_name(arg) for arg in op[1:-1]])+op[-1:])
                else:        
                    ret.append((name,)+op[1:])
            return tuple(ret)
        
        # local search in self.parent.child
        for op,img in self.parent.child.items():
            assert issubclass(type(op[0]), tuple), 'RoadImage.find_op: BUG: Invalid operation %s' % repr(op)
            if img is self:
                if not(raw):
                    # Replace method by method name in op
                    op = substitute_name(op)
                return ops+op
        if quiet:
            return ()
        raise ValueError('RoadImage.find_op: BUG: instance has parent, but cannot find op.')

    def __add_child__(self, ch, op, unlocked = False):
        """
        Internal method which adds ch as a child to self.
        In most cases, self becomes read only, but when ch is a numpy view (changes to self propagate
        automatically since the underlying data is the same), call with unlocked = True.
        """
        assert isinstance(op[0],tuple), 'RoadImage.__add_child: BUG: Trying to add old-style short op'
        assert not(isinstance(op[0][0],str)) , 'RoadImage.__add_child__: BUG: Storing string key'

        # Check if ch is already a child under some other operation
        # A fake crop operation may have been automatically assigned
        old_op = ()
        parent = self
        if not(ch.parent is None):
            # Look among ch.parent's children
            parent = ch.parent
            old_op = ch.find_op(parent = parent, quiet=True, raw=True)
        if not(old_op):
            # Look among self's children
            old_op = ch.find_op(parent = parent, quiet=True, raw=True)

        #for old_op, sibling in parent.child.items():
        #    if sibling is self: break
        if old_op:
            assert old_op[0][0].__name__ == 'crop' or old_op[0][0].__name__ == '__numpy_like__', \
                'RoadImage.__add_child__: BUG: returned instance is already a child. Conflict with %s.' % str(old_op)
            del parent.child[old_op] 
        # Make parent read-only: TODO in the future we would recompute the children automatically
        if self.flags.writeable != unlocked:
            self.flags.writeable = unlocked
        # Link ch to self
        ch.parent = self
        self.child[op] = ch
        return
    
    @static_vars(counter=0,dirname='./output_images/tracking')
    def track(self,state):
        if state.track != True and state.track!=state.counter: return
        import os
        from pathlib import Path

        dirname = os.path.normpath(RoadImage.track.dirname)
        if os.path.exists(dirname):
            if RoadImage.track.counter == 0:
                # Clear directory
                try:
                    p = Path(dirname)
                    kill_list = list(p.glob('*.jpg'))
                except:
                    pass
                else:
                    #print('Track deletes:',kill_list)
                    for f in kill_list:   f.unlink()
            RoadImage.track.counter += 1
            filename = os.path.join(dirname,'track%02d.jpg' % RoadImage.track.counter)
            self.save(filename, format='jpg')
        else:
            warnings.warn('RoadImage.track: dir %s does not exist.' % os.path.abspath(dir))
        return
    
    # Save to file
    @strict_accepts(object, str, str)
    def save(self, filename, *, format='png'):
        """
        Save to file using matplotlib.image. Default is PNG format.
        """
        if filename == self.filename:
            raise ValueError('RoadImage.save: attempt to save into original file %s.' % filename)
        nb_ch = RoadImage.image_channels(self)
        if not(nb_ch in [1,3,4]):
            raise ValueError('RoadImage.save: Can only save single channel, RGB or RGBA images.')
        flat = self.flatten()
        if flat.shape[0] != 1:
            raise ValueError('RoadImage.save: Can only save single images.')
        if flat.binary:
            if flat.dtype == np.float32 and flat.gradient:
                flat = flat.to_int()  # convert to int8
            if flat.dtype == np.int8:
                # Won't save signed
                flat = np.uint8(flat * 64)  # -1 -> 192, 1 -> 64, 0 -> 0
            elif flat.dtype == np.uint8 :
                flat = flat*255
        if flat.shape[3] == 1:
            mpimg.imsave(filename, flat.view(np.ndarray)[0,:,:,0], format=format, cmap='hot')
        else:
            mpimg.imsave(filename, flat[0], format=format)

    
    @strict_accepts(object, Axes, (str,None,Colormap), bool)
    @static_vars(__red_green_cmap__ = None)
    def show(self, axis, *, cmap=None, alpha=True, title=None):
        """
        Display image using matplotlib.pyplot.
        cmap is a colormap from matplotlib library. It is used only for single channel images and sensible defaults
        are provided.
        alpha can be set to False to ignore the alpha layer in RGBA images: pass "alpha=False".
        """
        # TODO : accept list/tuple of Axes and collection of N images with len(list)==N
        #        RoadImage.make_collection([img, img, img, img]).show(axes, ...)
        nb_ch = RoadImage.image_channels(self)
        if not(nb_ch in [1, 3, 4]):
            raise ValueError('RoadImage.show: Can only display single channel, RGB or RGBA images.')
        flat = self.flatten()
        if flat.shape[0] != 1:
            raise ValueError('RoadImage.show: Can only display single images.')

        if self.gradient:
            # Specific displays for gradients
            if nb_ch == 1:
                if cmap is None:
                    if RoadImage.show.__red_green_cmap__ is None:
                        colors = [(1, 0, 0), (0, 0, 0), (0, 1, 0)]  # R -> Black -> G
                        RoadImage.show.__red_green_cmap__ = LinearSegmentedColormap.from_list('RtoG', colors, N=256) 
                    cmap = RoadImage.show.__red_green_cmap__
                img = flat.to_float().view(np.ndarray)[0,:,:,0]
                axis.imshow(img, vmin=-1, vmax=1, cmap = cmap)
            else:
                if alpha:
                    img = flat[0].to_float()
                else:
                    img = flat[0,:,:,0:3].to_float()
                axis.imshow(np.abs(img), vmin=0, vmax=1)
        else:
            if nb_ch == 1:
                if cmap is None:
                    if   self.colorspace=='GRAY': cmap = 'gray'
                    elif self.colorspace=='RGB' : cmap = 'cubehelix'
                    elif self.colorspace=='HLS' or self.colorspace=='HSV' : cmap = 'hsv'
                    else:                          cmap = 'gnuplot2'
                # N.B. RoadImage does not allow the removal of the channels axis
                # therefore we have to view the data as a simple numpy array.
                img = flat.to_float().view(np.ndarray)[0,:,:,0]
                axis.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                if alpha:
                    img = flat[0].to_float()
                else:
                    img = flat[0,:,:,0:3].to_float()
                axis.imshow(img, vmin=0, vmax=1)
        if title:
            axis.set_title(title, fontsize=30)
                
    # Deep copy
    def copy(self):
        """
        In some cases, one needs a deep copy operation, which does nothing but copy the data.
        The returned RoadImage is not linked to self : copy is not an operation.
        find_op() will return an empty tuple, if called on a copy, but the crop area will usually
        be defined. 
        """
        # np.copy produces a numpy array, therefore view generates a blank RoadImage
        ret = np.copy(self).view(RoadImage)
        # Copy attributes
        ret.__array_finalize__(self)
        ret.binary = self.binary
        # The copy is not managed by the parent. The semantics of copy prohibit reuse of the same data.
        return ret

    # Flatten collection of images
    @generic_search(unlocked=True)
    def flatten(self):
        """
        Normalizes the shape of self to length 4. All the dimensions in which the collection of images is organised
        are flattened to a vector.
        In flattened form, shape is (nb_images,height,width,channels).
        This operation is always performed in place, without copying the data.
        """
        # Even an empty numpy array has one dimension of length zero
        assert self.shape[-1] == RoadImage.image_channels(self),\
            'RoadImage.flatten: BUG: last dimension must be channels. Shape is %s.' % repr(self.shape)

        # Test if already flat
        if self.ndim == 4:
            return self
            
        if self.ndim < 3:
            raise ValueError('RoadImage.flatten: RoadImage shape must be (height,width,channels).')
        
        if self.ndim > 3:
            # Flatten multiple dimensions before last three
            nb_img = np.prod(np.array(list(self.shape[:-3])))
            # np.prod is well-behaved, but will return a float (1.0) if array is empty
        else:
            nb_img = 1
        ret = self.reshape((nb_img,)+self.shape[-3:])
        #print('DEBUG(serial): ret =',ret.serial,' self=',self.serial) # serials increase by one

        assert not(ret.crop_area is None) , 'RoadImage.flatten: BUG: No crop area'
        return ret
        
    def channel(self,n):
        """
        Return a RoadImage with the same shape as self, but only one channel. 
        The returned RoadImage shares its underlying data with self.
        channel() is recorded as a channels() operation.
        """
        return self.channels(range(n,n+1))

    @strict_accepts(object,(int,range),(int,None),bool)
    @generic_search(unlocked=True)
    @flatten_collection
    def channels(self,n,m=None, *, isview=True):
        """
        Returns a RoadImage with the same shape as self, but only channels n to m.
        If m is larger than the number of channels in self, zeroed channels are added.
        For instance on a RGB image, channels(2,5) will place the blue channel of the source
        image in the red channel of the destination image, and will add two zeroed channels.
        By default x.channels(n) is a view on x data. Writing into channels() writes into x.
        It is mandatory to specify isview=False when adding new channels.
        """
        try:
            # See if it quacks
            rng = n
            n = rng.start
            m = rng.stop
            step = rng.step
        except AttributeError as e:
            try:
                if m is None: m = n+1
                step = 1
                rng = range(n,m,step)
            except TypeError as e:
                raise TypeError('RoadImage.channels: n and m must be two integers.')
            
        # Check for empty result
        if len([ch for ch in rng]) == 0:
            raise ValueError('RoadImage.channels: The selected range of channels must not be empty.')

        # Specific code
        if not(isview) or n<=-1 or n>=self.shape[3] or m>self.shape[3] or m<-1:
            # All the tests are needed because step can be negative to reverse the order of the channels
            if isview:
                raise ValueError('RoadImage.channels: Default behavior is not compatible with '+
                                 'the creation of new channels. Call with isview=False.')
            # Allocate new storage
            l = len([i for i in rng]) # Count channels to create
            if l==0:
                raise ValueError('RoadImage.channels: range(%d,%d,%d) results in zero channel image.'
                                 % (n,m,step))
            ret = np.zeros(shape=self.shape[0:3]+(l,), dtype=self.dtype).view(RoadImage)

            # Copy data
            cp = [i for i in rng if i>=0 and i < self.shape[3]]
            l = len(cp) # Count channels to copy
            # recompute n,m from cp.
            n = cp[0]
            m = cp[-1]
            if step > 0: m+=1
            else:        m-=1
            p = len([i for i in rng if i<0]) # Count prepended zero channels
            if l>0:
                # Same bug as below: self[:,:,:,n:m:step] could be empty if m<0
                if m<0:
                    ret[:,:,:,p:p+l] = self[:,:,:,n::step]  # Using cp or rng directly
                else:                                       # triggers an extra copy of self.
                    ret[:,:,:,p:p+l] = self[:,:,:,n:m:step]
        else:
            if (m<=n and step>0) or (m>=n and step<0):
                raise ValueError('RoadImage.channels: range(%d,%d,%d) results in zero channel image.'
                                 % (n , m , step))
            # There is a bug in numpy which makes it generate zero size arrays, collapsed in the
            # channels dimension, if m is explicitly listed as -1 (or some other negative value)
            # when step is negative. The equivalent n::step formulation works.
            if m<0:
                ret = self[:,:,:,n::step]
            else:
                ret = self[:,:,:,n:m:step]  # Main operation. Using rng triggers a copy.

        ret._inherit_attributes(self)
        return ret

    def rgb(self,n=0):
        """
        Shortcut for self.channels(0,3). Works to eliminate extraneous channels or to add new
        ones to single channel image. If channels are added the operation is not performed inplace,
        and the resulting image does not share data with its parent.
        Parameter n indicates the channel in the destination image, the source channel ends in.
        rgb() is recorded as a channels() operation.
        """
        try:
            ret = self.channels(range(0-n,3-n))
        except ValueError:
            ret = self.channels(range(0-n,3-n), isview=False)
        ret.colorspace='RGB'
        return ret
    
    # Operations which generate a new road image. Unlike the constructor and copy, those functions store handles to
    # newly created RoadImages in the self.children dictionary, and memorize the source image as parent.
    
    # Colorspace conversion
    @strict_accepts(object,str,bool)
    @generic_search()
    @flatten_collection
    def convert_color(self, cspace, *, inplace=False):
        """
        Conversion is done via RGB if self is not an RGB RoadImage.
        Only applicable to 3 channel images. See to_grayscale for grayscale conversion.
        inplace conversion creates a new object which shares the same buffer as self, and the data in the buffer
        is converted in place. 
        """
        # TODO: when uniformised for all inplace methods, transfer to decorator generic_search
        # NB: there is a variant np.empty_like(self[:,:,:,0:1]) when going from N to 1 channels.
        if inplace:
            ret = self
        else:
            ret = np.empty_like(self)

        # Input checking (beyond strict_accepts)
        if RoadImage.cspace_to_nb_channels(self.colorspace) != 3 or self.shape[3]!=3:
            raise ValueError('RoadImage.convert_color: This method only works on 3 channel images (e.g. RGB images).')
            
        if self.colorspace == cspace:
            # Already in requested colorspace
            return self

        for i , img in enumerate(self):
            if cspace == 'RGB':
                # "Inverse" conversion back to RGB
                cv2_code = RoadImage.cspace_to_cv2_inv(self.colorspace)
            else:
                cv2_code = RoadImage.cspace_to_cv2(cspace)
            cv2.cvtColor(img, cv2_code, dst = ret[i], dstCn = ret.shape[3])
        ret.colorspace = cspace              # update attributes of ret (and maybe self too if ret is self)

        return ret
    
    # Convert to grayscale (due to change from 3 to 1 channel, it cannot be done inplace)
    @strict_accepts(object)
    @generic_search()
    @flatten_collection
    def to_grayscale(self):
        if self.colorspace == 'GRAY' and self.shape[-1]==1:
            # Already grayscale
            return self
        assert RoadImage.cspace_to_nb_channels(self.colorspace) == 3 and self.shape[-1]==3 , \
               'RoadImage.to_grayscale: BUG: shape is inconsistent with colorspace.'

        ret = np.empty_like(self[:,:,:,0:1])
        ret.colorspace='GRAY'

        for i, img in enumerate(self):
            rgb = img.convert_color('RGB')  # conversion will be optimised out if possible. rgb child of self.
            ret[i] = np.expand_dims(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),-1)

        return ret
    
    # Format conversions
    @generic_search()
    def to_int(self):
        """
        Convert to integer.
        Gradient images are converted to [-127;+127] in int8 format.
        Other images are converted to [0:255] in uint8 format.
        """
        if self.dtype == np.uint8 or self.dtype == np.uint8:
            # Already int
            return self

        # Expensive input tests (magic constants allow some loss of accuracy and still round to 0 or 255)
        if np.max(self) > 1.00196:
            raise ValueError('RoadImage.to_int: maximum value of input is greater than one.')
        if self.gradient:
            if np.min(self) < -1.011:
                raise ValueError('RoadImage.to_int: minimum value of input is less than minus one.')
        else:
            if np.min(self) < -0.00196:
                raise ValueError('RoadImage.to_int: minimum value of input is less than zero.')

        if self.gradient:
            ret = np.empty_like(self, dtype=np.int8)
            scaling = 127.
        else:
            ret = np.empty_like(self, dtype=np.uint8)
            scaling = 255.
            
        if self.binary:
            ret[:] = self #.astype(np.int8)
        else:
            ret[:] = np.round(self * scaling) #.astype(np.int8)

        ret.binary = self.binary
        ret.gradient = self.gradient
        ret.warped = self.warped
        return ret

    @generic_search()
    def to_float(self):
        """
        Convert to floating point format (occurs automatically when taking gradients).
        Conversion to float assumes input integer values in range -128 to 127 for gradients, or else 0 to 255.
        Absolute value of gradients are converted like gradient, and will fit in [0;127] as integers, and [0;1] as fp.
        """
        if self.dtype == np.float32 or self.dtype == np.float64:
            # Already float
            return self

        if self.gradient:
            if self.dtype == np.uint8 and np.max(self) > 127:
                # Absolute value of gradient?
                warnings.warn('Warning: RoadImage.to_float: maximum value of gradient input is greater than 127.',
                              RuntimeWarning)
            scaling = 1./127.
        else:
            if self.dtype == np.int8:
                # Positive value stored as signed int
                warnings.warn('Warning: RoadImage.to_float: non negative quantity was stored in signed int.',
                              RuntimeWarning)
            scaling = 1./255.

        ret = np.empty_like(self, dtype=np.float32, subok=True)
        
        if self.binary:
            ret[:] = self #.astype(np.float32)
        else:
            ret[:] = scaling * self #.astype(np.float32)
        ret.binary = self.binary
        ret.gradient = self.gradient
        ret.warped = self.warped
        return ret

    # Remove camera distortion
    @strict_accepts(object, CameraCalibration)
    @generic_search()
    @flatten_collection
    def undistort(self, cal):
        """
        Uses a CameraCalibration instance to generate a new, undistorted image. Since it is a mapping operation
        it operates channel by channel.
        """
        if self.get_size() != cal.get_size():
            raise ValueError('RoadImage.undistort: CameraCalibration size does not match image size.')
        
        # There is no in place undistort, because the underlying opencv remap operation cannot do that.
        ret = np.empty_like(self)     # Creates ret as a RoadImage

        for index, img in enumerate(self):
            ret[index] = cal.undistort(img)
        ret.undistorted = True
        return ret
        
    # Resize image
    @strict_accepts(object, (int,tuple), (int,None))
    @generic_search()
    @flatten_collection
    def resize(self, w, *, h=None):
        """
        Resizes an image. Does not keep the initial aspect ratio since the operation can be used to
        prepare an image for displaying using rectangular pixels.
        """
        if type(w) is tuple:
            if not(h is None):
                raise TypeError('RoadImage.resize: h cannot be given when passing a tuple')
            h = w[1]
            w = w[0]
        elif h is None:
            # Keep aspect ratio
            h = (self.shape[1]*w)//self.shape[2]
            
        if w<=0 or h<=0:
            raise ValueError('RoadImage.resize: Both w and h must be strictly positive integers.')

        if w==self.shape[2] and h==self.shape[1]:
            # Because resize can be used to resize images read from files with different sizes,
            # it is important to always record the operation.
            # The slicing operation [:] creates a distinct RoadImage instance, which forces the
            # decorator to record the operation.
            return self[:]

        # Allocate space for result and choose method
        ret = np.empty(shape=(self.shape[0],h,w,self.shape[3]), dtype=self.dtype).view(RoadImage)
        method = cv2.INTER_CUBIC
        if h <= self.shape[1] and w <= self.shape[2]:
            method = cv2.INTER_AREA
            
        for index, img in enumerate(self):
            cv2.resize(img, dsize=(w,h), dst=ret[index], interpolation=method)

        return ret

    # Compute gradients for all channels
    @strict_accepts(object, (list, str), int, (float, int))
    def gradients(self, tasks, sobel_kernel=3, minmag=0.04):
        """
        Computes the gradients indicated by 'tasks' for each channel of self. Returns images with the same number of
        channels and dtype=float32, one per requested task.
        Because each task creates an independent child image, gradients returns a list of RoadImages rather than
        a single RoadImage containing a collection.
        Raw gradient images can reach values considerably higher than the maximum pixel value.
        8960 times (0 or 255) for Sobel 9 in x or y.
        640 times (0 or 255) for Sobel 7 in x or y.
        48 times (0 or 255) for Sobel 5 in x or y.
        4 times (0 or 255) for Sobel 3 in x or y.
        The returned images are scaled by the maximum theoretical value from the table above.
        tasks is a subset of [ 'x', 'y', 'absx', 'absy', 'angle', 'mag' ].        
        """
        from math import sqrt
        
        # Check content of tasks
        if isinstance(tasks, str):
            # Accept a single task in tasks (as a string)
            tasks = [tasks]
            return_as_list = False
        else:
            return_as_list = True
            
        if not(set(tasks).issubset({'x', 'y', 'absx', 'absy', 'mag', 'angle', 'absangle'})):
            raise ValueError("RoadImage.gradient: Allowed tasks are 'x','y','absx','absy','mag','angle' and 'absangle'.")
        if sobel_kernel % 2 == 0:
            raise ValueError('RoadImage.gradient: arg sobel_kernel must be an odd integer.')
        if sobel_kernel <= 0:
            raise ValueError('RoadImage.gradient: arg sobel_kernel must be strictly positive.')
        if minmag < 0:
            raise ValueError('RoadImage.gradient: arg minmag must be a non-negative floating point number.')
        
        flat = self.flatten()
        
        # Properly synthesize ops (very difficult to be fully compatible with @generic_search)
        ops = [ (tuple([ RoadImage.gradients, _make_hashable([task,sobel_kernel,minmag])]),) for task in tasks ]

        # Create new empty RoadImages for gradients, or recycle already computed ones
        grads = []
        for i,op in enumerate(ops):
            ret = self.find_child(op)
            if not(ret is None):
                grads.append(ret)
                tasks[i] = '_'     # Replace task by placeholder in list
            else:
                grads.append(np.empty_like(flat, dtype=np.float32))

        # Scaling factor
        if sobel_kernel <= 9:
            scalef = 1./RoadImage.SOBELMAX[sobel_kernel]
        else:
            # Compute scalef: convolve with an image with a single 1 in the middle
            single1 = np.zeros(shape=(2*sobel_kernel-1,2*sobel_kernel-1), dtype=np.float32)
            single1[sobel_kernel-1,sobel_kernel-1] = 1.0
            kernel = cv2.Sobel(single1, cv2.CV_32F, 1, 0, ksize=sobel_kernel)[::-1,::-1]
            scalef = 1./np.sum(b[:,k:])

        # Adjust scaling factor for maximum possible pixel value
        if self.dtype == np.uint8 and not(self.binary):
            scalef /= 255.
        elif self.dtype == np.int8 and not(self.binary):
            scalef /= 127.     # forget -128
        
        # Loop over channels
        for ch in range(RoadImage.image_channels(self)):
            # Loop over each channel in the collection
            # Calling flatten ensures that each channel is flat. Both flatten and channel generate low overhead views
            flat_ch = self.flatten().channel(ch)
            for img, gray in enumerate(flat_ch):
                # Loop over each image in the colleciton
                if set(tasks).intersection({'x', 'absx', 'mag', 'angle'}):
                    derx = scalef * cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize = sobel_kernel)
                    abs_derx = np.abs(derx)
                if set(tasks).intersection({'y', 'absy', 'mag', 'angle'}):
                    dery = scalef * cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize = sobel_kernel)
                    abs_dery = np.abs(dery)
                
                if 'x' in tasks:
                    index = tasks.index('x')
                    grads[index][img].channel(ch)[:] = np.expand_dims(derx,2)
    
                if 'absx' in tasks:
                    index = tasks.index('absx')
                    grads[index][img].channel(ch)[:] = np.expand_dims(abs_derx,2)
    
                if 'y' in tasks:
                    index = tasks.index('y')
                    grads[index][img].channel(ch)[:] = np.expand_dims(dery,2) 
    
                if 'absy' in tasks:
                    index = tasks.index('absy')
                    grads[index][img].channel(ch)[:] = np.expand_dims(abs_dery,2) 
    
                if ('mag' in tasks) or ('angle' in tasks):
                    # Calculate the magnitude (also used by 'angle' below)
                    grad = np.sqrt(abs_derx * abs_derx + abs_dery * abs_dery)/sqrt(2)
                    # Scale to 0-1
                    scaled_grad = grad/np.max(grad)
                    
                if 'mag' in tasks:
                    index = tasks.index('mag')
                    grads[index][img,:,:,ch] = grad
    
                if 'angle' in tasks:
                    index = tasks.index('angle')
                    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
                    aderx = np.copy(derx)
                    adery = np.copy(dery)
                    adermask=(derx<0)
                    aderx[adermask]=-derx[adermask]
                    adery[adermask]=-dery[adermask]
                    angle = np.arctan2(adery,aderx)
                    # Arctan2 returns value between -np.pi/2 and np.pi/2 which are scaled to [-1,1]
                    scaled = angle/(np.pi/2)
                    # Apply additional magnitude criterion, otherwise the angle is noisy
                    scaled[(scaled_grad < minmag)] = 0
                    grads[index][img,:,:,ch] = scaled

                if 'absangle' in tasks:
                    index = tasks.index('angle')
                    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
                    angle = np.arctan2(abs_dery, abs_derx)
                    # Arctan2 returns value between 0 and np.pi/2 which are scaled to [0,1]
                    scaled = angle/(np.pi/2)
                    # Apply additional magnitude criterion, otherwise the angle is noisy
                    scaled[(scaled_grad < minmag)] = 0
                    grads[index][img,:,:,ch] = scaled

        # TODO: Instead of handling ops here, we could have a decorated subfunction, using @generic_search
        # Which takes the same arguments as gradient, but a single task at a time, and grabs already
        # computed results in grads.
        #@generic_search()
        #def gradient(self, task, sobel_kernel, minmag):
            # Directly access list variables tasks and grads in 'gradients'
        #    i = tasks.index(task)
        #    return grads[i]

        # Return to original shape and set gradient flag
        for i,g in enumerate(grads):
            if tasks[i] != '_':
                # Only update the new ones
                # Set gradient flag for x and y which can take negative value
                g.gradient = (tasks[i] in ['x', 'y', 'angle'])
                grads[i] = g.reshape(self.shape)
                #TRYME: grads[i] = gradient(self, tasks[i], sobel_kernel = sobel_kernel, minmag = minmag)

        # Link to parent
        for i,(img,op) in enumerate(zip(grads,ops)):
            assert issubclass(type(img),RoadImage) , 'RoadImage.gradients: BUG: did not generate list of RoadImage!'
            if tasks[i] != '_':
                # Only link the new ones
                self.__add_child__(img, op)

        # When tasks is given as a string, return a single image.
        if len(grads)==1 and not(return_as_list):
            return grads[0]
        return grads

    @strict_accepts(object, bool, bool, bool)
    @generic_search()
    @flatten_collection
    def normalize(self, *, inplace=False, perchannel=False, perline=False):
        """
        Normalize amplitude of self so that the maximum pixel value is 1.0 (255 for uint8 encoded images).
        Gradients are normalized taking into account the absolute value of the gradient magnitude. The scaling
        factor is chosen so that the minimum gradient (negative) scales to -1, or the maximum gradient (positive) scales
        to 1. 
        Argument per should be 'image' (default), 'channel' or 'line'.
        Per channel normalization computes a scaling factor per channel. 
        Per image normalization computes a scaling factor per image, and all the channels are scaled by the same factor.
        It is best for RGB images and other colorspace with a linear relationship with RGB.
        Per line normalization computes a scaling factor per line.
        Per channel and per line can be combined.
        """
        # TODO: add a perimage=True argument and allow normalization at the scale of a collection
        # Is self already normalized?
        if self.binary:
            raise ValueError('RoadImage.normalize: Cannot apply normalize() to binary images.')

        maxi = np.maximum(self, 0-self)
        
        if perchannel:
            if perline:
                peak = maxi.max(axis=2, keepdims=True)   # one max per image, per line and per channel
            else:
                peak = maxi.max(axis=2, keepdims=True).max(axis=1, keepdims=True)    # one max per image and channel
        else:
            if perline:
                peak = maxi.max(axis=3, keepdims=True).max(axis=2, keepdims=True)  # one max per image and line
            else:
                peak = maxi.max(axis=3, keepdims=True).max(axis=2, keepdims=True).max(axis=1, keepdims=True)
            
        already_normalized = False
        if self.dtype == np.float32 or self.dtype == np.float64:
            if ((peak==1.0) | (peak==0.0)).all():
                already_normalized = True
            peak[np.nonzero(peak==0.0)]=1.0  # do not scale black lines
            scalef = 1.0/peak
        elif self.dtype == np.int8:
            if ((peak==127) | (peak==0)).all():
                already_normalized = True
            peak[np.nonzero(peak==0.0)]=1  # do not scale black lines
            scalef = 127.0/peak
        else:
            if self.dtype != np.uint8:
                raise ValueError('RoadImage.normalize: image dtype must be int8, uint8, float32 or float64.')
            if ((peak==255) | (peak==0)).all():
                already_normalized = True
            peak[np.nonzero(peak==0.0)]=1  # do not scale black lines
            scalef = 255.0/peak
        # Invariant: scalef defined unless already_normalized is True
        del peak, maxi

        if already_normalized:
            # Make copy unless inplace
            if inplace:
                return self
            ret = self.copy()
        else:
            if inplace:
                ret = self
            else:
                ret = np.empty_like(self)
            # Normalize
            ret[:] = scalef * self.astype(np.float32)
            del scalef

        return ret

    @strict_accepts(object, (float, int, np.ndarray), (float, int, np.ndarray), bool, bool)
    @generic_search()
    def threshold(self, *, mini=None, maxi=None, symgrad=True, inplace=False):
        """
        Generate a binary mask in uint8 format, or in the format of self if inplace is True.
        If symgrad is True and self.gradient is True, the mask will also include pixels with values
        between -maxi and -mini, of course assuming int8 or float dtype.
        mini and maxi are the thresholds. They are always expressed as a percentage of full scale.
        For int dtypes, fullscale is 255 for uint8 and 127 for int8, and for floats, fullscale is 1.0.
        This is consistent with normalize(self).
        mini and maxi must either:
        - be scalars
        - be vectors the same length as the number of channels: one threshold per channel
        - be a numpy array the same size as the image: shape=(height,width) . Operates as a mask for all channels.
        - generalize to the shape of self.flatten[1:] (the image size with channels): per pixel mini and maxi
        - generalize to self.shape
        It is therefore possible to apply thresholds and masks at the same time using per per pixel masks.
        The operator used is <= maxi and >= mini, therefore it is possible to let selected pixels pass.
        A binary gradient can have pixel values of 1, 0 or -1.
        """
        # Is self already binary?
        if self.binary:
            raise ValueError('RoadImage.threshold: Trying to threshold again a binary image.')

        # No thresholds?
        if (mini is None) and (maxi is None):
            raise ValueError('RoadImage.treshold: Specify mini=, maxi= or both.')
        
        # Ensure mini and maxi are iterables, even when supplied as scalars
        if isinstance(mini,float) or isinstance(mini,int):
            mini = np.array([mini], dtype=np.float32)
        if isinstance(maxi,float) or isinstance(maxi,int):
            maxi = np.array([maxi], dtype=np.float32)

        # Scale, cast and reshape mini according to self.dtype
        if not(mini is None):
            if np.any((mini < 0.0) | (mini > 1.0)):
                raise ValueError('RoadImage.threshold: Arg mini must be between 0.0 and 1.0 .')
            if self.dtype == np.int8:
                mini = np.round(mini*127.).astype(np.int8)
            elif self.dtype == np.uint8:
                mini = np.round(mini*255.).astype(np.uint8)
            else:
                if self.dtype != np.float32 and self.dtype != np.float64:
                    raise ValueError('RoadImage.threshold: image dtype must be int8, uint8, float32 or float64.')
            mini_shape = RoadImage.__match_shape__(mini.shape, self.shape)
            mini = mini.reshape(mini_shape)

        # Scale, cast and reshape maxi according to self.dtype
        if not(maxi is None):
            if np.any((maxi < 0.0) | (maxi > 1.0)):
                raise ValueError('RoadImage.threshold: Arg maxi must be between 0.0 and 1.0 .')
            if self.dtype == np.int8:
                maxi = np.round(maxi*127.).astype(np.int8)
            elif self.dtype == np.uint8:
                maxi = np.round(maxi*255.).astype(np.uint8)
            else:
                if self.dtype != np.float32 and self.dtype != np.float64:
                    raise ValueError('RoadImage.threshold: image dtype must be int8, uint8, float32 and float64.')
            maxi_shape = RoadImage.__match_shape__(maxi.shape, self.shape)
            maxi = maxi.reshape(maxi_shape)

        if inplace:
            data = self.copy()
            ret = self
            self[:] = 0
        else:
            data = self
            # Make new child and link parent to child
            if self.gradient and symgrad:
                dtype = np.int8
            else:
                dtype = np.uint8
            ret = np.zeros_like(self, dtype=dtype)
            #print('DEBUG(serials):',self.serial,'to',ret.serial)
            ret._inherit_attributes(self, binary_test=False)

        # Apply operation 
        if self.gradient:
            if mini is None:
                ret[(data <= maxi) & (data >= 0)] = 1
                if symgrad: ret[(data >= -maxi) & (data <= 0)] = -1
            elif maxi is None:
                ret[(data >= mini)] = 1
                if symgrad: ret[(data <= -mini)] = -1
            else:
                ret[((data >= mini) & (data <= maxi))] = 1
                if symgrad: ret[((data <= -mini) & (data >= -maxi))] = -1
        else:
            if mini is None:
                ret[(data <= maxi)] = 1
            elif maxi is None:
                ret[(data >= mini)] = 1
            else:
                ret[(data >= mini) & (data <= maxi)] = 1

        ret.binary = True
        return ret

    @strict_accepts(object, (np.ndarray), str)
    @generic_search()
    @flatten_collection
    def apply_mask(self, mask, op='and' , *, inplace=False):
        """
        Threshold does not work on binary images, and while make_collection().combine_masks() may work
        for binary images, it will trigger warnings if the mask is not derived from the masked image.
        Since usually masks come from other sources or files, which are fixed in the image processing
        pipeline, we want to record self as the parent, and not mask.
        If self is a collection, mask is applied to all the elements. 
        The method has no access to the shape of the collection within @flatten_collection. To avoid
        mistakes, the mask must be a single image or a collection of one image.
        A single channel mask will be replicated to match the number of channels in self.
        """
        if inplace:
            ret = self
        else:
            ret = self.copy()

        if not(issubclass(type(mask),RoadImage)) or not(mask.binary) or not(self.binary):
            raise TypeError('RoadImage.apply_mask: Image and mask must be binary RoadImages.')

        flatmask = mask.flatten()
        if flatmask.shape[3] != self.shape[3]:
            if flatmask.shape[3] != 1:
                raise ValueError('RoadImage.apply_mask: mask must have 1 channel or %d channels.'% self.shape[3])
            flatmask = np.repeat(flatmask, self.shape[3], axis=3)

        #zeroes = np.zeros(shape=(1,1,1,1))
        zeroes = np.zeros_like(ret[0])
        
        # The simplest implementation is to use make_collection, and combine_masks...
        # We avoid the warnings and copy the data.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ix, img in enumerate(self):
                ret[ix, (flatmask[0]==zeroes)]=0
                
        return ret

    @classmethod
    def warp_size(cls, cal, *, z, w=3.7, h, scale, curvrange):
        """
        Computes the size of the image needed to receive a warped image, using
        assumption regarding the curvature, and a defined scale in m/pixel.
        z and h are the parameters for the method cal.lane()
        Returns a dsize tuple in the correct format for cv2.warpPerspective(),
        and the position of the camera in pixels.
        """
        trapeze, rectangle, z_sol = cal.lane(z=z, w=w, h=h)
        lcurv, rcurv = curvrange
        # Impose minimum curvature which ensures that the start of the lane line is visible
        if lcurv > -0.001: lcurv = -0.001
        if rcurv <  0.001: rcurv =  0.001
        sx, sy = scale
        img_width = ( int(( 0.5 * z**2 * lcurv + rectangle[0][0]) / sx),
                      int(( 0.5 * z**2 * rcurv + rectangle[3][0]) / sx))
        img_height = int((z - z_sol) / sy)
        dsize = ( img_width[1]-img_width[0], img_height )

        return dsize, -img_width[0]

    def unwarp(self, *, z=70, h=0, scale=(.02,.1), curvrange=(-0.001, 0.001), cal=None):
        cal = self._retrieve_cal(cal)
        return self.warp(cal, z=z, h=h, scale=scale, curvrange=curvrange, _unwarp=True)[0]

    @generic_search()
    @flatten_collection
    def warp(self, cal, *, z=70, h=0, scale=(.02,.1), curvrange=(-0.001, 0.001), _unwarp=False):
        """
        Returns an image showing the road as seen from above and the location of the camera in 2D (in pixel).
        Imput image must be a camera image shape whose size matches cal.get_size().
        cal is a CameraCalibration instance. It must be parameterized correctly using 
        cal.set_camera_height(h) and cal.set_ahead(x,y,z).
        z and  h are passed to cal.lane() directly.
        scale is a tuple (sx,sy) giving the expected scale in meter/pixel of the output image.
        curvrange is the expected range of road curvature (deviations from a straight lane)
        and is used to compute an adequate width. It is also a tuple (curv_left, curv_right) and
        the first parameter should be negative, the second positive in 1/meter.
        """
        from itertools import chain
        
        # TODO: rectangle should have 2D space coordinates for a 3D mapping
        if _unwarp:
            if not(self.warped): raise ValueError('RoadImage.warp: Can only unwarp warped images.')
            offsets=(0,0)   # Unwarp always unwarps to original camera image size, then is cropped again.
            warp_from = self._retrieve_warp()
        else:
            if self.warped:      raise ValueError('RoadImage.warp: Image is already warped.')
            #if self.get_size() == cal.get_size():
            #    offsets=(0,0)
            #else:
            for p in chain([self], self.parents()):
                if p.get_size() == cal.get_size():
                    offsets = self.get_crop(p)[0]  # keep (x1,y1)
                    break
            else:
                raise ValueError('RoadImage.warp: Image size does not match cal.get_size().')
        
        sx, sy = scale

        trapeze, rectangle, z_sol = cal.lane(z=z,h=h)
        dsize, img_cx = RoadImage.warp_size(cal, z=z, h=h, scale=scale, curvrange=curvrange)
        origin = np.repeat(np.array([[img_cx, z/sy]], dtype=np.int), self.shape[0], axis=0)
        
        # Compute new rectangle coordinages in pixels, fitting inside dsize
        rect = np.array([ [x / sx + img_cx, dsize[1] - (y - z_sol) / sy ]
                          for x,y in rectangle ], dtype=np.float32)
        # Compute new trapeze coordinates taking into accoung crop
        trap = np.array(trapeze, dtype=np.float32) - np.array(offsets, dtype=np.float32)
        
        persp_mat = cv2.getPerspectiveTransform(trap, rect)
        
        # TODO: apply perspective transform to the bounding box of img (0,0),(w,0),(w,h),(0,h)
        #       compute bounding box of result, compute crop of this bounding box wrt dsize,
        #       offset trapeze coordinates (pixels) and return a smaller picture with crop_area set.
        #       May impact crop_area calculations since the parent is the warped image.
        #       May need to truncate image since crop_area extending beyond parent boundary is prohibited.
        
        # Storage for results
        if _unwarp:
            # if self.get_size() != dsize:
            #     raise ValueError('RoadImage.unwarp: Image size does not match size given by RoadImage.warp_size().')
            dsize = cal.get_size()
            
        # Current collection, original image size, current channel count
        ret = np.empty(shape=(self.shape[0],dsize[1],dsize[0],self.shape[3]), dtype=self.dtype).view(RoadImage)

        if _unwarp:
            for ix,img in enumerate(self):
                cv2.warpPerspective(img, persp_mat, dsize=dsize, dst=ret[ix],
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            # Compute the crop
            for p in warp_from.parents():
                if p.get_size() == cal.get_size():
                    cropping = warp_from.get_crop(p)
                    break
            else:
                raise ValueError('RoadImage.unwarp: Image size does not match cal.get_size().')
            ret = ret.crop(cropping)
        else:
            for ix,img in enumerate(self):
                cv2.warpPerspective(img, persp_mat, dsize=dsize, dst=ret[ix], flags=cv2.INTER_NEAREST)

        ret._inherit_attributes(self)   # Copy most attributes not affected by warping 
        ret.warped = not(_unwarp)
        return ret , origin

    @strict_accepts(object, tuple)
    @generic_search(unlocked=True)
    @flatten_collection
    def crop(self, area):
        """
        Returns a cropped image, same as using the [slice,slice] notation, but it accepts
        directly the output of self.get_crop().
        """
        (x1,y1),(x2,y2) = area
        if x2<=x1 or y2<=y1:
            raise ValueError('RoadImage.crop: crop area must not be empty.')
        return self[:,y1:y2,x1:x2,:]

    @strict_accepts(object, int, bool)
    @generic_search()
    @flatten_collection
    def despeckle(self, *, size=1, inplace=False):
        """
        Remove isolated dots from binary images.
        size indicates the size of the dots to eliminate in pixels.
        """
        if inplace:
            ret = self
        else:
            ret = self.copy()
            
        if not(self.binary):
            raise ValueError('RoadImage.despeckle: Image must be binary (e.g. from threshold).')
        
        zeroes = np.zeros_like(ret[0])
        
         # Opencv erode does not work on int8 images, and behaves as dilate for negative valued pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        # Process collection
        for ix, img in enumerate(self):
            if self.gradient:
                # Self is a signed binary image (+1, 0, -1)
                # we erode the mask, then we force zero everywhere the mask is zero.
                mask = np.abs(img.to_float())
                mask.gradient=False
            else:
                mask = img.copy()

            cv2.erode(mask, kernel, dst=mask, iterations=size)
            cv2.dilate(mask, kernel, dst=mask, iterations=size)
            ret[ix,(mask==zeroes)] = 0
            
        return ret
        
    @strict_accepts(object, int, bool)
    @generic_search()
    @flatten_collection
    def dilate(self, *, iterations=1, inplace=False):
        """
        Morphological dilatation of each channel independently.
        """
        if inplace:
            ret = self
        else:
            ret = self.copy()
            
        if not(self.binary):
            raise ValueError('RoadImage.despeckle: Image must be binary (e.g. from threshold).')

        ones = np.ones_like(ret[0])
        
        # Opencv erode does not work on int8 images, and behaves as dilate for negative valued pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        # Process collection
        for ix, img in enumerate(self):
            if self.gradient:
                # Self is a signed binary image (+1, 0, -1)
                # we erode the mask, then we force zero everywhere the mask is zero.
                mask = np.abs(img.to_float())
                mask.gradient=False
            else:
                mask = img.copy()

            cv2.dilate(mask, kernel, dst=mask, iterations=iterations)
            ret[ix,(mask==ones)] = 1
            
        return ret
        
    @strict_accepts(object, int, bool)
    @generic_search()
    @flatten_collection
    def erode(self, *, iterations=1, inplace=False):
        """
        Morphological erosion of each channel independently.
        """
        if inplace:
            ret = self
        else:
            ret = self.copy()
            
        if not(self.binary):
            raise ValueError('RoadImage.despeckle: Image must be binary (e.g. from threshold).')
        
        zeroes = np.zeros_like(ret[0])

        # Opencv erode does not work on int8 images, and behaves as dilate for negative valued pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        # Process collection
        for ix, img in enumerate(self):
            if self.gradient:
                # Self is a signed binary image (+1, 0, -1)
                # we erode the mask, then we force zero everywhere the mask is zero.
                mask = np.abs(img.to_float())
                mask.gradient=False
            else:
                mask = img.copy()

            cv2.erode(mask, kernel, dst=mask, iterations=iterations)
            ret[ix,(mask==zeroes)] = 0
            
        return ret
        
    @strict_accepts(object, (int, float, str), bool)
    @generic_search()
    def combine_masks(self, op, *, perchannel=True):
        """
        Accepts a collection of N masks, each one either single channel or multichannel,
        and combines them together using P in N rule. If P=1, an 'OR' is performed,
        if P=N, an 'AND' is performed. For intermediate values, a majority vote takes place.
        Parameter op can be an integer P, or keywords 'or' and 'and' as strings, or a float percentage
        between 0 and 1 (Caution: 1 is equivalent to 'or', and 1. is equivalent to 'and').
        Per channel is the default mode of operation, where distinct combine operations take
        place on each channel, returning an image with the same number of channels as the
        collection. If per channel is False, the collection is processed like a collection of
        single channel images and a single channel image is returned.
        On gradient masks, -1 gradients compensate +1 gradients in other images. The resulting
        pixel is -1 if the total is <=-P, and +1 if the total is >=P.
        """
        flat = self.flatten()
        if not(self.binary):
            raise ValueError('RoadImage.combine_masks: Image collection must be binary.')
        if perchannel:   nbch=1
        else:            nbch=flat.shape[3]
        nbimg = flat.shape[0]
        if op=='and':  op = nbimg * nbch  
        elif op=='or': op = 1
        elif type(op) is str:
            raise ValueError("RoadImage.combine_masks: op must be 'or', 'and', or a number.")
        elif type(op) is float:
            if op > 1. or op < 0.:
                raise ValueError("RoadImage.combine_masks: expressing a percentage, float op must be between 0. and 1.")
            op = int(round(op * nbimg * nbch))
            if op == 0:
                op=1   # Force minimum 1, otherwise pixels at zero after the sum are undecided in gradient mode
                warnings.warn("RoadImage.combine_masks: percentage for success is too low", RuntimeWarning)

        if op < 1 or op > nbimg * nbch:
            raise ValueError("RoadImage.combine_masks: op must be an integer between 1 and the number of images.")
        
        if perchannel:
            ret = np.zeros_like(flat[0:1,:,:,:])
            raw = flat.sum(axis=0, keepdims=True)
        else:
            ret = np.zeros_like(flat[0:1,:,:,0:1])
            ret.colorspace='GRAY'
            raw = flat.sum(axis=(0,3), keepdims=True)
        
        ret[(raw >= op)] = 1
        if self.gradient:
            ret[(raw <= -op)] = -1
            
        ret.binary=True
        ret.gradient = self.gradient
        ret.warped = self.warped
        
        # Return a collection of the same ndim, but all collection dimensions collapsed to shape=1.
        sh = RoadImage.__match_shape__(ret.shape[:-1], self.shape)
        sh = sh[:-1] + (ret.shape[-1],)

        return ret.reshape(sh)

    @strict_accepts(object,int,bool,bool,bool)
    @generic_search()
    @flatten_collection
    def integrate(self, *, ksize=100, invertx=False, inverty=False, inplace=False):
        """
        Works channel by channel on signed gradient images, which can be binary or not.
        It uses the sign(d)/d² integration kernel to integrate by convolution in 2D.
        The result is technically a gradient but not a binary image.
        Two 1D passes are applied since the convolution is separable.
        The result is not normalized.
        invertx reverses the sign of the kernel during the x pass.
        """
        # TODO: some kind of local automatic balancing of + and -
        if inplace:
            ret = self
        else:
            ret = np.empty_like(self)
            
        if not(self.gradient):
            raise ValueError('RoadImage.integrate: Operates on signed gradient images.')
        if not(self.dtype == np.float32 or self.dtype == np.float64):
            raise ValueError('RoadImage.integrate: Operates on float32 or float64 images. Use to_float().')
        if not(ksize <= self.shape[1] and ksize <= self.shape[2]):
            raise ValueError('RoadImage.integrate: Kernel size must be less than image width and height.')

        # Some monotonous shape decreasing to -1, for instance linear
        #kernel = np.array([ -2.*i/ksize for i in range(ksize//2) ], dtype=np.float32)
        kernel = np.array([ -1. for i in range(ksize//2) ], dtype=np.float32)
        # Antisymmetric duplication to ksize length
        kernel = np.concatenate([kernel, -kernel[::-1]])
        
        for ix, img in enumerate(self):
            # X-pass
            for iy, line in enumerate(img):
                # Operate line by line
                for ic in range(self.shape[3]):
                    if invertx:  l = np.convolve(line[:,ic], -kernel, mode='same')
                    else:        l = np.convolve(line[:,ic], kernel, mode='same')
                    ret[ix,iy,:,ic]=l
            # Y-pass
            for iy in range(self.shape[2]):
                # Operate column by column
                col = img[:,iy,:]
                for ic in range(self.shape[3]):
                    if inverty:  l = np.convolve(col[:,ic], -kernel, mode='same')
                    else:        l = np.convolve(col[:,ic], kernel, mode='same')
                    ret[ix,:,iy,ic]+=l
        ret.binary = False        
        return ret

    @generic_search()
    def invert(self):
        """
        Does the very simple operation of changing the sign of signed gradients.
        Doing it with numpy makes it lose its binary quality.
        """
        if not(self.gradient):
            raise ValueError('RoadImage.invert: Image must be a signed gradient (gradient x or gradient y).') 
        ret = np.empty_like(self, subok=True)
        ret[:] = -self
        ret.binary=True
        return ret
    
    def __slice__(self):
        """
        Placeholder method used to trace lignage between images when a = b[slice]
        but a is not a crop of b.
        """
        pass

    def __numpy_like__(self):
        """
        Placeholder method used to trace operations done with numpy: zeros_like, ones_like, empty_like...
        """
        pass

    def _retrieve_warp(self):
        """
        Retrieve parent which was warped to self.
        """
        if not(self.warped):
            raise ValueError('RoadImage._retrieve_warp: only applicable to warped images, their likes and their crops.')
        warp_ops=None
        for p in self.parents():
            warp_op = RoadImage.select_ops(self.find_op(parent=p, raw=False, quiet=False), ['warp'])
            if warp_op: break
        else:
            raise ValueError('RoadImage._retrieve_warp: No warp operation found in image history (BUG?).')
        return p
    
    def _retrieve_cal(self, cal=None):
        """
        Retrieve cal associated with nearest of wrap, undistort, in the ancestry of
        the image.
        """
        warp_undist_ops=None
        for p in self.parents():
            if p.parent is None:
                warp_undist_ops = RoadImage.select_ops(self.find_op(parent=p, raw=False, quiet=False),
                                                       ['warp', 'undistort'])
                break

        if warp_undist_ops: nearest = warp_undist_ops[-1]
        else: return cal
        # Two possibilities depending on whether op was called like this: op(cal), or like this: op(cal=cal).
        calis=[]
        calobj=[]
        for arg in nearest[1:]:
            if arg[0] == dict:
                for a in arg[1:]:
                    if issubclass(type(a[1]),CameraCalibration):
                        if a[0]=='cal':      calis.append(a[1])
                        else:                calobj.append(a[1])
            else:
                for a in arg:
                    if issubclass(type(a), CameraCalibration):  calobj.append(a)
        if calis:    my_cal = calis[0]
        elif calobj: my_cal = calobj[0]
        else:        my_cal = None
        if cal is None:
            if my_cal is None:
                raise ValueError('RoadImage._retrieve_cal: No CameraCalibration associated with this image. Pass arg cal.')
            else:
                return my_cal
        elif my_cal is None:
             warnings.warn('RoadImage._retrieve_cal: No CameraCalibration associated with this image. Using arg cal.') 
           
        return cal
    
    @strict_accepts(object, tuple, str, (tuple,None))
    @generic_search()
    @flatten_collection
    def curves(self, key, *, dir='x', width=0.12, origin, scale, wfunc=None, minpts=150, sfunc=None, cal=None, **kwargs):
        """
        Extract curves from self.
        key is a tuple. The result will be a geometry referenced by key.

        If dir='y', key may fetch a function giving y as a function of x, and broadly horizontal lines
        will be researched. Conversely if dir='x', key may fetch a function giving x as a function of 'y',
        and curves will look for broadly vertical lines (i.e. lane lines on a warped image).
        if there is no apriori knowledge of the lines ('one',) can be used to search on the right of the
        car, and Line.blend(key1=('zero',),key2=('one',),w1=1,w2=-1) to search on the left at roughly
        1 meter from the car centerline.

        width indicates the width of the lines we are looking for in meters.
        origin is the location of the camera in x,y (in pixels and usually off the bottom of self).
        scale is (sx,sy) the scale of the warped self in meters/pixel.
        wfunc is a weight function, which will be used to reduce the importance of points situated far from the 
        hint.
        minpts is the number of points curves will try to pass to np.polyfit.
        sfunc is a scoring function which will be passed the key and the x and y of each contributing
        partial solution, and which should return a score between 0 and 1. It can be used to assess how
        far the current solution is from an ideal line start from the previous frame for instance.

        cal is the camera calibration instance which was used to warp self. It will be retrieved from
        self's history in most cases, but you may need to pass it if self is a synthetic test image for instance.
        cal will only be used if curves cannot find the instance used for warping, and a warning will be printed.

        curves paints itself on self using predefined Line attributes.
        """
        # TODO: weight based on euclidian distance between key and solution...
        
        # When a curve of a given width is extracted from a warped image, there is an interval that
        # (dX/dY) must stay in for a given X,Y,Z location from the camera, or else the line becomes invisible.
        # The ideal is lines which radiate from the camera and appear vertical in perspective.
        def deriv_bounds(cal, X,Y,Z, width):
            """
            For a Line geometry given in camera axes (X=0,Y=0,Z>0 is the optical axis), this function
            gives the acceptable bounds of the derivative of that Line wrt Z, dX/dZ(X,Y,Z), which make
            the detection by the camera of that object plausible.
            If the test is failed on distant parts of the Line geometry, it means that the description is
            invalid at that distance (may have detected a bigger object, or maybe it's an extrapolation).
            """
            # Plausible slope variation from ideal slope, function of height above road Y, distance Z
            # camera resolution given by focal_length() and width of painted line.
            if Y is None: Y = cal.camera_height
            dxdz = np.tan(np.arcsin(np.minimum(np.ones_like(Z),cal.focal_length('y')*abs(Y)*width/ Z**2)))
            # ideal slope
            ideal = X/Z
            return (ideal-dxdz, ideal+dxdz)

        cal = self._retrieve_cal(cal)
        
        cnt, h, w, ch = self.shape

        # Per pixel values are fetched only from the first image... subgrids too: 0 is hardcoded in both places. 
        assert cnt==1, 'RoadImage.curves: BUG! Does not work on multi-image collections.'

        # The grid is a stipple pattern made of two slices. The x and y arrays of nonzero pixels are
        # concatenated before passing them to polyfit.
        sx,sy = scale
        stepx = max(int(width/sx),2)
        stepy = max(int(width/sy),2)
        startx = stepx//2
        starty = stepy//2
        iterx  = stepx - startx
        itery  = stepy - starty
        
        # test if wfunc uses pixel values wfunc(x,y,val)
        wrap_w=False
        if wfunc:
            try:
                w = float(wfunc(0,0,self[0,0,0]))
            except (TypeError,ValueError):
                # Problem with number of arguments or type of returned value
                try:
                    w = float(wfunc(0,0))
                except (TypeError,ValueError) as e:
                    raise TypeError('RoadImage.curves: arg wfunc must accept 2 or 3 arrays/scalars of the same length.')
            else:
                wrap_w=True
        
        # origin and accumulators
        x0,y0 = origin
        resultkey = ('result',current_thread())
        acckey = ('acc',current_thread())  # thread safe key for storage in shared self.lines
        self.lines.copy(self.lines.zero,acckey)  # use thread local key in shared lines dictionary
        acc_w = 0
        acc_z_max = 0
        acc_wsco = 0
        acc_csco = 0
        acc_dsco = 0

        # Small variant of iterator ensures that the inner loop is on Y after the eventual swap of X and Y.
        if dir=='x':
            iters = ( (i,j) for j in range(itery) for i in range(stepx) )
        elif dir=='y':
            iters = ( (i,j) for i in range(iterx) for j in range(stepy) )
        else:
            raise ValueError("RoadImage.curve: argument dir must be either 'x' or 'y'.")
        
        # subgrids
        for i,j in iters:
            sub = self.view(np.ndarray)[0,j::stepy,i::stepx]
            y1,x1,_ = np.nonzero(sub)  # pixels appear once for each channel they are =1 in (weight by channels)
            if wrap_w:
                val1 = sub[y1,x1]  # May be RGB pixels or not, but always vectors of length 1 or more
            sub = self.view(np.ndarray)[0,j+starty::stepy,i+startx::stepx]
            y2,x2,_ = np.nonzero(sub)
            if wrap_w:
                val2 = sub[y2,x2]
            # Wrap values with weight function
            if wrap_w:    weight_func = partial(wfunc, val=np.concatenate([val1,val2],axis=0))
            else:         weight_func = wfunc
                
            # Compute physical coordinates
            X1 = (x1*stepx+i-x0)*sx
            Y1 = (y0-y1*stepy-j)*sy
            X2 = (x2*stepx+i+startx-x0)*sx
            Y2 = (y0-y2*stepy-j-starty)*sy
            # Concatenate
            X = np.concatenate([X1,X2], axis=0)
            Y = np.concatenate([Y1,Y2], axis=0)
            # Test if there are enough pixels
            if len(X)==0: continue # No fitting algo works with zero data
            
            # Pass everything to self.lines.fit(), which does y data conditioning using key,
            # solves, and records the geometry in its private format in self.lines.
            if dir=='x':  # verticals X=f(Y)
                X,Y = Y,X
            # From now on, we always solve Y=f(X)
            try:
                self.lines.copy(key,resultkey)
            except KeyError:
                key = self.lines.zero
                self.lines.copy(key, resultkey)

            try: # Only now will we know if X and Y contain enough data
                self.lines.fit(resultkey, X, Y, wfunc=weight_func, **kwargs)
            except ValueError:
                # lines.fit tells us our data is junk
                continue # to next stipple pattern

            # Rate solution (refY cannot be done out of the loop because X changes)
            refY = self.lines.eval(key, z=X)
            newY = self.lines.eval(resultkey, z=X)
            ## weight function
            if wrap_w:
                # if weight function uses pixel values, we must go back to image and fetch those values
                if dir=='x': # X=f(Y)
                    y = np.round(y0 - X/sy).astype(np.intp)
                    refx = np.round(refY/sx + x0).astype(np.intp)
                    newx = np.round(newY/sx + x0).astype(np.intp)
                    refval = self[0, y, refx]
                    newval = self[0, y, newx]
                else:       # Y=f(X)
                    x = np.round(X/sx + x0).astype(np.intp)
                    refy = np.round(y0 - refY/sy).astype(np.intp)
                    newy = np.round(y0 - newY/sy).astype(np.intp)
                    refval = self[0, refy, x]
                    newval = self[0, rewy, x]
                refW = wfunc(X, np.zeros_like(X), refval)
                newW = wfunc(X, newY-refY, newval)
            elif wfunc:
                refW = wfunc(X, np.zeros_like(X))
                newW = wfunc(X, newY-refY)
            else:
                refW = np.ones_like(X)
                newW = refW
            refw = sum(refW)
            neww = sum(newW)
                
            weight_score = neww/refw

            # Detect loss of lines in image at long distance
            if dir=='x':
                z_max = np.max(X)
                if wfunc:
                    # We can refine: find z at 90% of total weight neww
                    for z in range(int(z_max),0,-1):
                        if sum(newW[(X <= z)])<= 0.90*neww :
                            z_max = z/0.90 # Increase geometrically (weights tend to spread out much more)
                            break
                    else:
                        warnings.warn("RoadImage.curves: images may be black?", RuntimeWarning)
            else:
                z_max = np.max(Y)
                warnings.warn("RoadImage.curves: z_max implementation for dir='y' does not yet use wfunc.", RuntimeWarning)
                # Very crude implementation: we should check if Y leaves the [0,z_max] interval when x is in X, and
                # if the curve escapes through the top, we should propose a larger z_max.
                
            ## deriv_bounds test
            # The score is the proportion of the curve which complies with the constraint, stopping
            deriv = self.lines.eval( resultkey, z=X, der=1 )
            # sort points : near to far, away from 0, and calculate derivative bounds
            if dir=='x':
                index = np.argsort(abs(Y)+1000*X)
                derbounds = deriv_bounds(cal, Y, None, X, width)
            else:
                index = np.argsort(abs(X)+1000*Y)
                derbounds = deriv_bounds(cal, X, None, Y, width)
                # By construction, horizontal curves cannot pass this test with null derivative
                # even if at that location deriv_bounds allows nearly +/-90° angle. Nudge slightly.
                deriv[(abs(deriv)<=np.finfo(np.float32).eps)]=np.finfo(np.float32).eps
                deriv = 1./deriv
            lbounds = ((deriv >= derbounds[0]) & (deriv <= derbounds[1]))
            if np.all(lbounds): bounds_score = 1.
            else:               bounds_score = (1+np.argmin(lbounds[index]))/len(lbounds)

            ## custom scoring function
            cust_score = 1  # No influence, default
            if sfunc:
                try:
                    cust_score = sfunc(self.lines,key,resultkey,dir,X,newY)
                    if cust_score < 0.1 : cust_score=0.1  # do not formally eliminate solutions
                    elif cust_score > 1 : cust_score=1    # cap influence
                except Exception as e:
                    warnings.warn('RoadImage.curves: custom scoring function exception ignored.\n%s' % str(e),
                                  RuntimeWarning)
            
            # Debug
            #print('DEBUG: p(%d,%d)='% (i,j), self.lines.stats(resultkey,'poly'))
            #print('DEBUG: p(%d,%d) scores: cust=%0.2f  weight=%0.2f  derbounds=%0.2f'
            #      % (i,j,cust_score, weight_score, bounds_score))
            # Accumulate solution
            weight = cust_score * weight_score * bounds_score
            acc_w += weight
            acc_z_max += z_max*weight
            acc_csco += cust_score*weight
            acc_dsco += bounds_score*weight
            acc_wsco += weight_score*weight
            self.lines.blend(acckey,key1=acckey,key2=resultkey,
                             op='wsum', w1=1, w2=weight)
            # Draw solution...

        # No usable pixels
        if acc_w == 0:
            raise ValueError('RoadImage.curves: No useable pixels. Is image black?')
        
        # Loop on subgrids finished - normalize result and save under arg key.
        self.lines.blend(key, key1=self.lines.zero, key2=acckey,
                         op='wsum', w1=0, w2=1/acc_w)
        self.lines.set(key,'csco',acc_csco/acc_w)
        self.lines.set(key,'dsco',acc_dsco/acc_w)
        self.lines.set(key,'wsco',acc_wsco/acc_w)
        self.lines.set(key,'zmax',acc_z_max/acc_w)
        self.lines.delete(resultkey)
        self.lines.delete(acckey)
        return np.array([acc_z_max/acc_w])
        
    # The next section contains specialized operators for autonomous road vehicles
    # - extract_lines does image processing to robustly identify pixels belonging to
    #   lane lines on the picture.
    # - find_lines extracts the lines, then searches the beginning of left and right
    #   lane lines, and returns them. Various algorithms are then available to
    #   model the lines.
    # - find_window_centroids is the function from the course: given a list starting point
    #   it creates a mask covering the pixels of that line only.
    # - fit_poly returns the coefficients of a polynomial fitting the pixels in an image.
    # - draw_lane accepts two polynomial descriptions of lane lines and draws a mask
    #   covering the lane over some defined distance.

    @strict_accepts(object, CameraCalibration, tuple)
    @generic_search()
    @flatten_collection
    def extract_lines(self, *, lrmask, mask=None):
        """
        Does color thresholding and gradients analysis in multiple colorspaces, to robustly
        extract the pixels belonging to lane lines on an image.
        
        lanekey is the key in self.lines of a curve representing the center of the lane.
        mask is an optional mask. Pixels excluded by the mask will be blackened out.
        """
        # The lrmask is used to select pixels in operations when detect preferentially one
        # of the lines. x,y gradient ops do this, as well as [TODO] signed gradient orientation ops.

        class State(object):
            pass
        
        state = getattr(self,'elstate',None)
        if state is None:
            state = State()
            self.elstate = state
            state.mask = None
            state.lrmask = None
            state.cspace = 'HLS'
            state.mini = [0.0627, 0.762, 0.75]
            state.maxi = [0.0863, 1.0,   1.0]
            
        if state.mask is None:
            if mask is None:
                state.mask = np.ones_like(self.channel(0), dtype=np.uint8)
                state.mask.binary=True
            else:
                # TODO: could allow TRAPEZE coordinates as well.
                state.mask=mask
        mask = state.mask

        # Check mask
        if mask.get_size()!=self.get_size() or not(mask.binary) or mask.shape[-1]!=1:
            raise ValueError('RoadImage.extract_lines: mask must be binary, single channel, same size as self.')
        
        # Outer product of per channel thresholds with per pixel mask.
        # The small offset on minimask ensure that it is larger than maximask on masked pixels
        mini = np.tensordot(mask, [state.mini], axes=([-1],[0]))+0.0001
        maxi = np.tensordot(mask, [state.maxi], axes=([-1],[0]))

        if lrmask is None:  lrmask = state.lrmask
        # Check lrmask
        if lrmask is None or lrmask.get_size()!=self.get_size() or not(lrmask.binary) or lrmask.shape[-1]!=1:
            raise ValueError('RoadImage.extract_lines: lrmask must be binary, single channel, same size as self.')
        state.lrmask = lrmask
        
        img = self

        # The code below is by no means the unique way of getting those images...
        # Colors...
        imgcol = img.convert_color(state.cspace).to_float()
        # Mask and threshold in one operation using 3 channel, per pixel masks
        imgcol.threshold(mini=mini, maxi=maxi, inplace=True)
        imgcol.dilate(iterations=1, inplace=True)
        
        # Gradients...
        img = img.convert_color('HLS')
        (a,m,x,y) = img.gradients(['angle','mag','x','y'], sobel_kernel=9)
        a.threshold(mini=np.array([0.266,0.269,0.265]),
                    maxi=np.array([0.64,0.661,0.685]), inplace=True)\
         .despeckle(size=2)
        m.normalize(perchannel=True, inplace=True)\
         .threshold(mini=np.array([0.352,0.298,0.327]),
                    maxi=np.array([0.623,0.707,0.705]),inplace=True)\
         .despeckle(size=2)
        x.normalize(perchannel=True, perline=True, inplace=True)\
         .threshold(mini=np.array([0.304,0.3,0.302]),
                    maxi=np.array([0.687,0.707,0.707]),inplace=True)
        y.normalize(perchannel=True, perline=True, inplace=True)\
         .threshold(mini=np.array([0.28,0.278,0.282]),
                    maxi=np.array([0.684,0.707,0.706]),inplace=True)
        
        # Special processing for left lane line
        # TODO: Cancel negative part of signed 'a'
        gleft = RoadImage.make_collection([x,y], concat=True)\
                         .combine_masks('or', perchannel=False)\
                         .integrate(ksize=20, invertx=False)
        # Cancel right side and negative part
        #gleft[(lrmask==1) | (gleft<0)] = 0
        gleft[(gleft<0)] = 0
        gleft.normalize(inplace=True, perline=True)
        gleft.threshold(mini=0.3, inplace=True)
        #gleft.despeckle(size=1, inplace=True)

        # Special processing for right lane line
        # TODO: idem above
        gright = RoadImage.make_collection([x.invert(),y], concat=True)\
                         .combine_masks('or', perchannel=False)\
                         .integrate(ksize=20, invertx=True)
        # Cancel left side and negative part
        #gright[(lrmask==0) | (gright<0)] = 0
        gright[(gright<0)] = 0
        gright.normalize(inplace=True, perline=True)
        gright.threshold(mini=0.3, inplace=True)
        #gright.despeckle(size=1, inplace=True)

        gam = RoadImage.make_collection([a,m], concat=True)\
                       .combine_masks('or', perchannel=False)\
                       .dilate(iterations=3, inplace=True).erode(iterations=5, inplace=True)
        #g   = RoadImage.make_collection([gam,gxy], concat=True)\
        #               .combine_masks('or') #.despeckle(size=2, inplace=True)
        g   = RoadImage.make_collection([gam,gleft,gright], concat=True)\
                       .combine_masks(2)
        gr  = g.apply_mask(mask)

        final = RoadImage\
                .make_collection([imgcol.channel(0), imgcol.channel(1), imgcol.channel(2), gr],
                                 concat=True)\
                .combine_masks(2, perchannel=False)  #op=2 or 1

        # import matplotlib.pyplot as plt
        # figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) \
        #     = plt.subplots(nrows=4, ncols=2, figsize=(20,26))
        
        # gxy = RoadImage.make_collection([gleft, gright], concat=True).combine_masks('or')
        # img.show(ax1, title='Original image')
        # g.show(ax2, title='Result of gradients')
        # gxy.show(ax3, title='Gradients x and y (not used)')
        # gam.show(ax4, title='Gradients a and m')
        # gr.show(ax5, title='Masked gradients')
        # imgcol.show(ax6, title='colors masked')
        # gleft.show(ax7, title='Integrated gradient (left)')
        # gright.show(ax8, title='Integrated gradient (right)')

        return final
        
    @generic_search()
    @flatten_collection
    @static_vars(state=None)
    def find_lines(self, cal, *, z=70, h=0, scale=(.02,.1)):
        """
        Finds the lane lines on an image which has been corrected for distortion.
        """
        # Do not work on collections (other than singletons)
        if self.shape[0]!=1:
            raise ValueError('RoadImage.find_lines: Must process one image at a time.')
        if self.colorspace!='RGB' or self.undistorted==False:
            raise ValueError('RoadImage.find_lines: Must input undistorted, cropped if necessary, RGB image only.')
        
        from scipy import signal
        from scipy.stats import norm

        def reinit_lines(lines=['left','right']):
            if 'left' in lines:
                state.lines.copy(('minusone',),curK('left'))
            if 'right' in lines:
                state.lines.copy(('one',),curK('right'))
            if 'farleft' in lines:
                state.lines.blend(curK('farleft'), key1=curK('left'), key2=state.lines.one,
                                  op='wsum', w1=1, w2=-state.lanew)
            if 'farright' in lines:
                state.lines.blend(curK('farright'), key1=curK('right'), key2=state.lines.one,
                                  op='wsum', w1=1, w2=state.lanew)
            for li in lines:
                state.lines.set(curK(li),'eval',0.001)
                state.lines.set(curK(li),'order',0)
            return
                
        def scoring(lines, key_in, key_out, dir, x, y, der=0):
            """
            Scoring of partial solutions in RoadImage.curves.
            This one penalizes curves which stray far away at short range.
            """
            # TODO: survive when key_in does not exist in lines
            Z0=5 # meters
            NORMSCALE=1.5
            x_in  = lines.eval(key_in, z=Z0, der=der)
            x_out1 = lines.eval(key_out,z=Z0, der=der)
            w1 = norm.pdf(x_out1, loc=x_in, scale=NORMSCALE)/norm.pdf(0,scale=NORMSCALE)
            x_out2 = lines.eval(key_in, z=x, der=der)
            w2 = np.exp(-np.sum((x_out2-y)**2)/np.ptp(x)**2)
            #print('scoring:',x_in, x_out1, w1, w2)
            return w1*w2

        def weight(x,y,val,x0=0.5,z0=100):
            """
            Weights of points according to location and value
            """
            # y is horizontal from reference, x is distance from camera
            # val is RGB, and binary image data can be in any layer.
            return (np.sum(val,axis=-1)+0.01)*np.exp(-x**2/z0**2)*np.exp(-y**2/x0**2)

        MIN_EVAL = 0.001
        def eval(key, warpimage, origin, scale, x0, z0):
            """
            Returns an evaluation of key as a lane line in a warped image.
            Buffer must be the same size as self, since state will be used to translate coordinates.
            The evaluation is the ratio of the number of lit pixels under the line rendering to the
            total number of lit pixels in the line rendering.
            """
            # TODO : should give a bonus for well contrasted sides...
            assert warpimage.ndim==4, 'RoadImage.find_lines.eval: BUG in warpimage format / must be collection.'
                
            _,y0 = origin  # Caution: weight() param is also named x0
            sx,sy = scale
            
            linebuf = np.zeros_like(warpimage.channel(0), dtype=np.uint8)[0]
            # Save and restore z_max and dens (which are taken into account by draw)
            z_max, dens = state.lines.stats(key, 'zmax','dens')
            state.lines.set(key,'dens',1)
            state.lines.set(key,'zmax',linebuf.shape[0]*sy)
            # Draw a thin line to limit the number of points
            try:
                state.lines.draw(key, linebuf, origin=origin, scale=scale, color=[255], width=2*sx)
            except ValueError:
                warnings.warn('RoadImage.find_lines.eval: Line.draw failed. Geometry may be invalid.')
                #import pdb; pdb.set_trace()
                ev = MIN_EVAL
            else:
                linebuf.track(state)
                if dens is None:  state.lines.unset(key,'dens')
                else:             state.lines.set(key,'dens',dens)
                if z_max is None: state.lines.unset(key,'zmax')
                else:             state.lines.set(key,'zmax',z_max)

                y, x, _  = np.nonzero(linebuf)
                val = warpimage[0,y,x]
                # Convert coordinates
                Y = (y0-y)*sy
                # Apply weight (with itself as a reference)
                w = weight(Y,0,val,x0,z0)
                wref = weight(Y,0,1,x0,z0)

                ev = max(MIN_EVAL, np.sum(w)/np.sum(wref))

            state.lines.set(key,'eval',float(ev))
            return ev

        def stageK(key,stage):
            return (key[0],state.counter,stage)

        def estimates(lines, keepn=6, update=True, bestfirst=True, order=4):
            """
            Looks into state.lines to find recent line detections, and produce a list of current initial estimates
            sorted by decreasing stage of preparation.
            keepn is the number of recent geometries to use to make a new estimate.
            If update is True, the current values curK(line)  will be updated to reflect the returned values,
            otherwise the result will be associated to specific keys which will be returned.
            Returned keys are normally sorted by decreasing evaluation. If bestfirst is False, the returned keys
            will be in the order of argument lines instead.
            Returned estimates will be simplified to order=order. The default is 4, meaning that they are not simplified.
            """
            lines_dict = dict()
            recent = state.counter - keepn
            linelib= state.lines
            tmpkey = ('RoadImage.estimates',)
            to_delete = []
            
            for K in linelib:
                if len(K) < 2: continue
                li, counter = K[:2]
                if not(isinstance(counter,int) and li in lines) : continue
                if counter > state.counter:
                    to_delete.append(K)
                    continue
                if counter < recent:
                    if counter < state.counter - 60: to_delete.append(K)
                    continue
                try:
                    lines_dict[li].append(K)
                except KeyError:
                    lines_dict[li] = [ K ]

            for K in to_delete:
                linelib.delete(K)
                
            wght=lambda ev_,o_: ev_ #*(1+o_)
            target = {}
            for li in lines:
                if update:   target[li] = curK(li)
                else:        target[li] = ('tmp'+li,)

            ZTAN=6  # distance at which simplified estimates tangent old geometries 
            
            for li in lines:
                tgt = target[li]
                try:
                    it = iter(lines_dict[li])   # KeyError if lines_dict[li] was never created.
                except KeyError:
                    # Put zero in the pool as a stage 0 estimate
                    #linelib.copy(state.lines.zero, tgt)
                    print('Initializing '+li+' line.')
                    reinit_lines(lines=[li])
                    linelib.set(curK(li),'eval',MIN_EVAL)  # 0.001 is a very low eval. Avoid 0 to make sure total is never 0.
                    if not(update): linelib.copy(curK(li),tgt)
                else:
                    K = next(it)          # there is at last one element in the list.
                    totev = linelib.stats(K, 'eval')
                    toto = linelib.stats(K, 'order')
                    total = wght(totev,toto)
                    linelib.copy(K, tmpkey)
                    linelib.tangent(tmpkey, z=ZTAN, order=order)
                    linelib.blend(tgt, op='wsum', key1=linelib.zero, key2=tmpkey, w1=1, w2=total)
                    count = 1
                    for K in it:
                        o_ = min(2,linelib.stats(K, 'order'))
                        ev_ = linelib.stats(K, 'eval')
                        totev = totev + ev_
                        toto = toto + o_
                        w_ = wght(ev_,o_)
                        total = total + w_
                        count += 1
                        linelib.copy(K, tmpkey)
                        linelib.tangent(tmpkey, z=ZTAN, order=order)  # Take tangent at 10 m order 2 solution
                        linelib.blend(tgt, op='wsum', key1=tgt, key2=tmpkey, w1=1, w2=w_)

                    linelib.blend(tgt, op='wsum', key1=tgt, key2=state.lines.zero, w1=1/total, w2=0)
                    linelib.set(tgt,'eval', totev/count) # Average eval
                    linelib.set(tgt,'order', toto/count) # Fractional, just to know... doesn't seem to hurt.

            ret = [target[li] for li in lines]
            if not(bestfirst): return ret
            return sorted(ret, key=lambda K: linelib.stats(K,'eval'), reverse=True)
        
        def log(K,m=''):
            if state.journal:
                with open(state.journal,'a') as f:
                    poly = state.lines.stats(K,'poly')
                    data = state.lines.stats(K,'eval','zmax','order')
                    data = list(data)
                    if len(K)>1:data.append(K[1])
                    else:   data.append(0)
                    poly = [ str(k) for k in poly ]
                    data = [ str(k) for k in data ]
                    msg = ';'.join([data[3],data[0],data[1],data[2],poly[0],poly[1],poly[2],poly[3],poly[4]])
    
                    f.write('"'+m+str(K)+'";'+msg.replace('.',',')+'\n')
            return
        
        curK = lambda li_ : (li_, state.counter)

        # Each image in a sequence is typically a new RoadImage instance which enters the pipeline.
        # The State object carries some information across from one image to the next.
        class State(object):
            pass

        # Priority 1 is from self, priority 2 is static storage
        state = getattr(self, 'find_lines_state', RoadImage.find_lines.state)
        if state is None:
            state = State()
            RoadImage.find_lines.state = state
            
        if not(hasattr(state,'lines')):
            # state may have been found, only because the caller initialized a few tunable parameters
            # We detect this by checking for 'lines', which is not a tunable. It's not critical though
            # because we only print a message out. The initialisation is done attribute by attribute
            # using the _state lambda function below.
            print('Initialization.')
            tunables = { 'zmax', 'keepnext', 'keepdraw', 'journal', 'track' }
            tuned = tunables.intersection(set(dir(state)))
            for attr in tuned:
                print('   ',attr,'=',getattr(state,attr,None))

        _state = lambda attr, val: setattr(state,attr,getattr(state,attr,val))
        _state('lz',z)
        _state('rz',z)
        _state('delta',(0,0))
        _state('curv',0)       # lane curvature
        _state('lanew',3.7)    # lane width
        _state('cal',cal)      # camera calibration
        _state('counter',0)    # frame counter
        _state('lines',Line.Factory())    
        # Tunables
        _state('zmax',z)       # Via z argument
        _state('keepnext', 10) # Number of past frames to average to _stateiate sync on current one
        _state('keepdraw',  4) # Number of past frames to average for drawing
        _state('journal', None)# Logging
        _state('track', None)  # Image logging
        # Filters
        _b, _a = signal.butter(4, 1/10)
        _state('curv_b',_b)
        _state('curv_a',_a)
        _state('curv_zi', signal.lfilter_zi(state.curv_b, state.curv_a)*state.curv)
        _b, _a = signal.butter(4, 1/10)  # distinct instances
        _state('lzmax_b',_b)
        _state('lzmax_a',_a)
        _state('lzmax_zi', signal.lfilter_zi(state.lzmax_b, state.lzmax_a)*state.lz)
        _b, _a = signal.butter(4, 1/10)  # distinct instances
        _state('rzmax_b',_b)
        _state('rzmax_a',_a)
        _state('rzmax_zi', signal.lfilter_zi(state.rzmax_b, state.rzmax_a)*state.rz)

        # Initial lines constants
        state.lines.blend(('minusone',), key1=state.lines.zero, 
                          key2=('one',), op='wsum', w1=0, w2=-1)
        #reinit_lines()

        # Store in self too. self.lines is how we get information out of find_lines().
        self.find_lines_state = state
        #self.lines = state.lines

        # Colors
        GREEN = np.array([ 0, 229, 153, 50 ])
        RED = np.array([ 255, 25, 0, 128 ])
        BLACK = [0,0,0,255]  # Used to erase already identified lanes
        MAGENTA = [192,0,192,64]
        
        # Do the work
        _, aheady = cal.get_ahead(state.curv)
        crop = slice(aheady,656)
        img = self[:,crop]
        #img.lines = state.lines   # Recall previous lines
        
        # Theoretical lane appearance and distance to "start of lane".
        trapeze, _, z_sol = cal.lane(z=z,h=h)
        
        # Blend left and right to get median line

        if state.lines.exist(curK('left'), curK('right')):
            state.lines.blend( curK('lane'), key1=curK('left'), key2=curK('right'), op='wavg', w2=0.5 )

            # Get and filter z_max
            lz = state.lines.stats(curK('left'),'zmax')
            rz = state.lines.stats(curK('right'),'zmax')
            if lz == None: lz = state.lz
            if rz == None: rz = state.rz
            
            # Measure curvature
            curvature = np.mean(state.lines.curvature( curK('lane') , z=np.array([10,20,30])))

            # Low pass data
            curvature, state.curv_zi = signal.lfilter(state.curv_b, state.curv_a, [curvature], zi=state.curv_zi)
            curvature = curvature[0]
            lz, state.lzmax_zi = signal.lfilter(state.lzmax_b, state.lzmax_a, [lz], zi=state.lzmax_zi)
            lz = lz[0]
            state.lz = lz
            rz, state.rzmax_zi = signal.lfilter(state.rzmax_b, state.rzmax_a, [rz], zi=state.rzmax_zi)
            rz = rz[0]
            state.rz = rz
        else:
            curvature = 0
            lz = state.lz
            rz = state.rz
        state.curv = curvature

        # Increment image counter in sequence (important to do it AFTER curvature above since curK uses it)
        state.counter += 1
        
        curvrange = (curvature, curvature)
        #_warpo   = lambda _img: _img.warp(  state.cal, z=state.zmax, h=h, scale=scale, curvrange=curvrange)
        _warpomethod = partialmethod(RoadImage.warp, state.cal, z=state.zmax, h=h,
                                     scale=scale, curvrange=curvrange)
        setattr(RoadImage,'_warpo',_warpomethod)
        #_unwarp = lambda _img: _img.unwarp(z=state.zmax, h=h, scale=scale, curvrange=curvrange, cal=state.cal)
        _unwarpmethod = partialmethod(RoadImage.unwarp, z=state.zmax, h=h,
                                      scale=scale, curvrange=curvrange, cal=state.cal)
        setattr(RoadImage,'_unwarpm',_unwarpmethod)
        #_warpo  = lambda _img: _img._warpomethod()
        _warp   = lambda _img: _img._warpo()[0]
        _unwarp = lambda _img: _img._unwarpm()

        # Image preprocessing
        # -------------------
        gray = img.to_grayscale()
        gray = gray.threshold(mini = 0.5)

        overlay, orig = gray._warpo() #_warpo(gray)
        origin = (orig[0,0],orig[0,1])
        _, _,w,ch = overlay.shape

        sums = np.sum(overlay.astype(np.int),axis=(-2,-1))
        overlay = overlay.to_float()

        if np.max(sums[0,:]) > 80:
            # Light colored pavement ahead
            # Use pixel processing from the project walkthrough
            hsV = img.convert_color('HSV').channel(2).threshold(mini=50/255)
            hlS = img.convert_color('HLS').channel(2).threshold(mini=100/255)
            gradx,grady = img.to_grayscale().gradients(tasks=['absx','absy'])
            gradx.normalize(inplace=True).threshold(mini=12/255, inplace=True)
            grady.normalize(inplace=True).threshold(mini=25/255, inplace=True)
            color=RoadImage.make_collection([hsV, hlS], concat=True).combine_masks('and')
            gxy = RoadImage.make_collection([gradx.to_int(), grady.to_int()], concat=True).combine_masks('and')
            layer = RoadImage.make_collection([gxy, color], concat=True).combine_masks('or').despeckle()
                
            #layer,_ = layer.warp(cal, scale=scale)
            layer,_ = layer.warp(state.cal, z=state.zmax, scale=scale, curvrange=curvrange)
            #layer = _warp(layer)
            layer = layer.to_float()
            overlay.channel(0)[(sums>100),:,0] = 0 # Clear out bad contrast zones with simple algorithm
            overlay.channel(0)[(layer>0.5)]=1

        # Get initial estimate into curK(line)
        keys = estimates(['left','right'], keepn=state.keepnext, update=True)
        for K in keys: log(K,'E')
        orderlist = [0, 1, 2, 4]  # Polynomial curve orders we use
        
        # Define realistic masks according to estimate known eval
        for K in keys:
            overlay.track(state)

            ev = state.lines.stats(K,'eval')
            order = state.lines.stats(K,'order')   # Average order, generally fractional
            maxstages = 5  # gives some options to repeat a few stages
            orderit = filter(lambda x: x>order-1, orderlist)
            order = next(orderit)
            initial_eval = 0


            while True: # Loop on stages
                # define mask width in meters based on estimated eval of estimated line
       
                if ev < 0.05:                # stage 0
                    #maskw = (state.lanew - 0.8)*2
                    maskw, order, der, x0, z0, nextev = 3, 0, 0, 2.5, 15, 0.15
                elif ev < 0.15 or order <= 1: # stage 1 - dashed lines have inherently lower evals
                    maskw, order, der, x0, z0, nextev = 2, 1, 1, 1, 30, 0.25
                elif ev < 0.25 or order <= 2: # stage 2
                    maskw, order, der, x0, z0, nextev = 1, 2, 0, 0.5, 100, 0.7
                else:                        # stage 3
                    maskw, order, der, x0, z0, nextev = 1, 4, 0, 0.5, 1000, 0.7

                # Make a copy and render estimated line as a mask (extending zmax)
                mask = np.zeros_like(overlay.channel(0), dtype=np.uint8)
                mask.binary=True
                
                z_max = state.lines.stats(K,'zmax')
                if z_max: state.lines.set(K,'zmax', 2*z_max)
                state.lines.draw(K, mask[0], origin=origin, scale=scale, color=[1], width=maskw)
                mask.track(state)
                if z_max: state.lines.set(K,'zmax', z_max)
                # Further clear points which are not in the overlay
                mask[overlay==0] = 0
                
                
                #print("mask %3.1f m ="% maskw, state.lines.stats(K,'poly'))
                mask.track(state)
                
                # Shorthand
                my_eval=partial(eval, warpimage=mask, origin=origin, scale=scale)
                # Reassess K: loop until eval tops out
                ev = my_eval(K, x0=x0, z0=z0)
                best_eval = ev
                state.lines.copy(K,curK('save'))
                if ev > initial_eval: initial_eval = ev
                improving = -1   # number of loops ending with a positive test in the while
                # Mostly useful before the first detection (stage 0):
                if order == 0: maxiters = 10
                else:          maxiters = 3    
                focus_k  = (1/x0)**(1/maxiters)  # the 1st '1' is x0 = 1 meter at stage 1.

                while ev<0.05 or ev>=best_eval*0.9:  # Convergence : K changes at each call of curves
                    try:
                        z_max = mask.curves(K, dir='x', order=order,
                                            origin=origin, scale=scale,
                                            sfunc=partial(scoring, der=der),
                                            wfunc=partial(weight,val=1,x0=x0,z0=z0))
                    except ValueError: # Catch image is black? exception
                        mask.track(state)
                        reinit_lines(lines=[K[0]])
                        order = 0
                    else:
                        state.lines.set(K,'order',order)
                    ev = my_eval(K, x0=x0, z0=z0)
                    log(K,'K')
                    
                    if ev > best_eval*1.01: # Minimal improvement +1% 
                        improving += 1     
                        best_eval = ev
                        state.lines.copy(K,curK('save'))
                    elif ev > nextev:
                        # We are not improving any more, but we have reached the ev level for the next higher order.
                        if ev < best_eval*0.95:
                            # Restore best solution only if the current one is significantly worse
                            state.lines.copy(curK('save'),K)
                        break
                    else:
                        maxiters  -= 1     # else consume one "free" iteration
                        x0 *= focus_k      # but focus, since distant noise can otherwise keep us just beside a line
                        z0 /= focus_k**2   # and look farther ahead to try to catch dashes, rather than noise
                    #print('.',end='')
                    if maxiters == 0: break
                else:
                    # This exit is taken only if ev falls lower than best_eval*0.9
                    # Restore best solution for this stage
                    state.lines.copy(curK('save'),K)
                    log(K,'S')
                state.lines.delete(curK('save'))
 
                ev, csco, wsco, order, z_max = state.lines.stats(K,'eval','csco','wsco','order','zmax')

                # print(" %s : poly ="% str(K), state.lines.stats(K,'poly'))
                #if csco!=None:
                #    print("    zmax = %4.1f  scores: eval = %4.2f  cust = %4.2f  weights = %4.2f"
                #          % (z_max, ev, csco, wsco))

                # eval cannot increase forever: its maximum value is 1.
                if ev > initial_eval * 1.1:
                    # Fast convergence
                    #print('+',end='')
                    #print("Fast convergence!")
                    initial_eval = ev    # ensures we do not loop forever on continue
                    continue   # same order and stage
                elif order == 4:
                    #print("Synchronized.")
                    # Erase line from overlay: it helps curves focus on the remaining, less visible lines
                    state.lines.draw(K,overlay[0], origin=origin, scale=scale, color=BLACK, 
                                     width=maskw)
                    break
                elif ev > nextev:
                    # more or less converged: next stage
                    pass
                elif ev <= initial_eval * 0.8:
                    # Collapse
                    print("Lost sync! ev %6.4f < %6.4f initial_eval" % (ev, initial_eval))
                    # eliminate solution from pool
                    state.lines.delete(K)
                    break
                else: # Stagnation
                    maxstages -= 1
                    if maxstages==0:
                        print("Stuck at order %d. Moving on." % order)
                    break
                # Possibly other quality considerations here...
                try:
                    order = next(orderit)
                except StopIteration:
                    print('StopIteration BUG : order = %d.'%order)
                    break
            # Loop to increase order

        # Detection finished. In case of failures to detect, we generate solutions from past solutions
        
        key1, key2 = estimates(['left','right'], keepn=state.keepdraw, update=False, bestfirst=False)
        log(key1,'W')
        log(key2,'W')
        
        lz = state.lines.stats(key1,'zmax')
        if lz is None and state.lz > 5: state.lz -= 1
        lz = state.lz
        
        rz = state.lines.stats(key2,'zmax')
        if rz is None and state.rz > 5: state.rz -= 1
        rz = state.rz
            
        # Measure lane width
        distk = np.array([1,2,3])
        # eval at low multiples of z_sol since farther out there are perspective errors 
        l_center = np.mean(state.lines.eval( key1,  z=z_sol*distk ))
        r_center = np.mean(state.lines.eval( key2, z=z_sol*distk ))
        lane_width = r_center - l_center
        if lane_width > 3.5 and lane_width < 3.9 :
            # state.lanew is used in detection: shouldn't wander too far off.
            state.lanew = lane_width
        # reassess l_center and r_center based on official lane width
        corrlw = state.lanew/lane_width
        l_center *= corrlw
        r_center *= corrlw
            
        # TODO: reassess short term camera height based on perceiveed short-distance lane_width.
        # Camera height is defined by the car's geometry, but it can have short term variations
        # around an average value due to the car's suspensions. Long term and large variations of
        # perceived lane width are real changes of the lane width. Note that car height depends
        # also on loading conditions (and suspensions settings on some luxury cars).
        # Implementation:
        #     lane_width --low-pass filter--> real_lw
        #     lane_wdith - real_lw --> ac_lw
        #     cam height = f(ac_lw)

        self[:,crop] = img
        backgnd = self
        green = GREEN.tolist()
        cv2.putText(backgnd[0],"%0.2f" % (-l_center), (int(640 - 100*state.lanew/2), 55),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=green, thickness=2)
        cv2.putText(backgnd[0],"%0.2f" % r_center, (int(640 + 100*state.lanew/2 - 135), 55),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=green, thickness=2)
        cv2.line(backgnd[0],(int(640 - 100*state.lanew/2), 70),(int(640 + 100*state.lanew/2), 70),
                 color=green, thickness=2)
        cv2.circle(backgnd[0],(int(640 - 100*(state.lanew/2 + l_center)), 70), 10, color=green, thickness=2)
        if curvature > 0.0005:
            cv2.putText(backgnd[0],"R=%4d m" % int(1/curvature), (860,250),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=green[:3], thickness=2)
        elif curvature < -0.0005:
            cv2.putText(backgnd[0],"R=%4d m" % int(-1/curvature), (100,250),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=green[:3], thickness=2)

        # Draw on background image
        kz = min(lz,rz)
        if kz > state.zmax: kz = state.zmax
        if kz < 10: kz = 10
        kz = (kz-10) / (state.zmax-10)      # ratio of achieved z to desired z (full red at 10 m)
        color = (GREEN * kz + RED * (1-kz)).astype(np.uint8)

        state.lines.set(key1,'zmax',lz)
        state.lines.set(key2,'zmax',rz)
        state.lines.draw_area( key1, key2, backgnd[0], origin=origin, scale=scale, color=color,
                              warp=_warp, unwarp=_unwarp)
        # Draw lane lines
        state.lines.draw(key1,  backgnd[0], color=[255,255,0,200], origin=origin, scale=scale,
                         warp=_warp, unwarp=_unwarp)
        state.lines.draw(key2, backgnd[0], color=[255,255,0,200], origin=origin, scale=scale,
                         warp=_warp, unwarp=_unwarp)

        return self
    
    def centroids(self, *, x, lanew, scale):
        """
        'centroids' is called by find_lines() on a single image of a flattened collection.
        The returned image is RGB, with the warped B&W image as blue, the left lane line as green, 
        and the right lane line as red.
        self is an image.
        x is the initial x coordinate of the lane line to extract.
        line is a handle to an instance of class Line, which must be updated by the method.
        lanew is the lane width on self, expressed in pixels, used as a scaling factor for the centroid.
        scale is the scale of the warped image in m/pixel, stored as (sx,sy). 
        """
        # self is warped image
        assert self.warped, "RoadImage.centroids: Call this method via 'find_lines()'."
        sx, sy = scale
        h = self.shape[0]

        # The course does centroids layer 4 m long
        nb_layer = int(h*sy/5)
        window_height = h // nb_layer
        window_width = lanew * 50 // 640
        margin = lanew * 100 // 640
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        ret = np.zeros_like(self)  # Single channel image

        # Ancillary subroutine to draw masks
        # ret, self, h, window_height, window_width are read from the enclosing function environment.
        def window_mask(center,level):
            ret.view(np.ndarray)[int(h-(level+1)*window_height):int(h-level*window_height),
                max(0,int(center-window_width/2)):min(int(center+window_width/2), self.shape[1])] = 1

        # Trace first centroid
        window_mask(x, 0)
        delta = 0
        deltax = 0
        
        for level in range(1,nb_layer):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(self[int(h-(level+1)*window_height):int(h-level*window_height),:],
                                 axis=(0,2))
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is
            # at right side of window, not center of window
            offset = window_width/2
            min_index = int(max(x+offset-margin,0))
            max_index = int(min(x+offset+margin,self.shape[1]))
            if len(conv_signal[min_index:max_index])==0: newx = 0
            else: newx = np.argmax(conv_signal[min_index:max_index])
            if newx == 0:
                x=x+deltax  # No detection, argmax returns 0, the value is invalid
            else:
                newx += min_index-offset
                deltax = 0.5*(delta + newx - x) # sliding average for delta
                x = newx
            # Draw what we found for that layer
            window_mask(x, level)

        return ret

    @generic_search()
    @flatten_collection
    @static_vars(state=None)
    def find_cars(self, clf_file, vis_heatmap=False):
        """
        This method finds the cars on a road image, and draws boxes around them.
        """
        # Do not work on collections (other than singletons)
        if self.shape[0]!=1:
            raise ValueError('RoadImage.find_cars: Must process one image at a time.')
        if self.undistorted==False:
            raise ValueError('RoadImage.find_cars: Must input undistorted, cropped if necessary.')

        from math import hypot
        
        # Sub-functions
        ###############
        
        # Def a function to return HOG features and visualisation
        def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
            from skimage.feature import hog

            # Call with two outputs if vis = True
            if vis == True:
                features, hog_image = hog(img, orientations=orient,
                                          pixels_per_cell=(pix_per_cell, pix_per_cell),
                                          cells_per_block=(cell_per_block, cell_per_block),
                                           transform_sqrt=False,
                                          visualise=True, feature_vector=feature_vec)
                return features, hog_image
            # Otherwise call with one output
            else:
                features = hog(img, orientations=orient,
                               pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block),
                               transform_sqrt=False,
                               visualise=False, feature_vector=feature_vec)
                return features

        # Def a func to compute binned color features
        def bin_spatial(img, size=(32,32)):
            color1 = cv2.resize(img[:,:,0], size).ravel()
            color2 = cv2.resize(img[:,:,1], size).ravel()
            color3 = cv2.resize(img[:,:,2], size).ravel()
            return np.hstack((color1, color2, color3))     # dimshuffle...

        # Def a function to compute color histogram features
        def color_hist(img, nbins=32):
            # Compute the histogram of the color channels separately
            channel1_hist = np.histogram(img[:,:,0], bins=nbins)
            channel2_hist = np.histogram(img[:,:,1], bins=nbins)
            channel3_hist = np.histogram(img[:,:,2], bins=nbins)
            # Concatenate the histograms into a single feature vector
            hist_features = np.concatenate([channel1_hist[0], 
                                            channel2_hist[0], 
                                            channel3_hist[0]])
            # Return the individual histograms, bin_centers and feature vector
            return hist_features
        
        # Define a single function that can extract features using hog sub-sampling and make predictions
        def find_at_scale(img, area, scale, blockmap, svc, X_scaler, cspace,
                          orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

            xstart,ystart = area[0]
            xstop, ystop  = area[1]
            #import pdb; pdb.set_trace()
            img_tosearch = img[ystart:ystop,xstart:xstop,:].to_float()
            blockmap.binary = True
            blockmap = blockmap[ystart:ystop,xstart:xstop].to_float()  # for resize...
           
            ctrans_tosearch = img_tosearch.convert_color(cspace).copy() #'HSV'/ YCC'
            if scale != 1:
                imshape = ctrans_tosearch.shape
                xspan = int(round(float(imshape[1]/scale)))
                yspan = int(round(float(imshape[0]/scale)))
                ctrans_tosearch = ctrans_tosearch.resize((xspan, yspan))
                blockmap = blockmap.resize((xspan, yspan))
                blockmap.binary = False
                blockmap = blockmap.threshold(mini=0.5)
                

            ch1 = ctrans_tosearch.view(np.ndarray)[:,:,0]
            ch2 = ctrans_tosearch.view(np.ndarray)[:,:,1]
            ch3 = ctrans_tosearch.view(np.ndarray)[:,:,2]

            # Define blocks and steps as above
            width, height = ctrans_tosearch.get_size()
            #nxblocks = (width // pix_per_cell) - cell_per_block + 1
            #nyblocks = (height // pix_per_cell) - cell_per_block + 1 
            nfeat_per_block = orient*cell_per_block**2

            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            half_window = window**2 // 2
            nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            #nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
            #nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
            ncells_per_window = window // pix_per_cell
            nxcells = width // pix_per_cell
            nycells = height // pix_per_cell
            nxsteps = (nxcells - ncells_per_window) // cell_per_block + 1
            nysteps = (nycells - ncells_per_window) // cell_per_block + 1

            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

            bboxes = []

            #print(nxsteps*nysteps,'windows at scale %4.2f.'%scale)
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Test patch in blockmap and avoid matching already blocked areas
                    intersection = np.sum(blockmap[ytop:ytop+window, xleft:xleft+window])
                    if intersection > half_window: continue
                    
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    # Extract the image patch
                    subimg  = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    hist_features = color_hist(subimg, nbins=hist_bins)

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                    test_prediction = svc.predict(test_features)

                    if test_prediction == 1:
                        xleft_draw = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        match = ((xleft_draw+xstart, ytop_draw+ystart),
                                 (xleft_draw+win_draw+xstart,ytop_draw+win_draw+ystart)) 
                        bboxes.append(match)
                        if state.track == state.counter:
                            print('match: scale=%4.2f  %s  %.0f'% (scale,str(match),intersection))
            return bboxes
        
        # Define a function which takes a heatmap, adds val to each pixel contained
        # in each box (so multiple times in some cases), get a derived map with a threshold
        # and using scipy's label function, return individual boxes for each object.
        def apply_heat(heatmap, blockmap, scale, offset, bboxes, val, doboxes=True):
            from scipy.ndimage.measurements import label
            
            x0,y0 = offset
            for bb in bboxes:
                x1,y1 = bb[0]
                x2,y2 = bb[1]
                x1 = int(round((x1-x0)/(scale*16)))
                y1 = int(round((y1-y0)/(scale*16)))
                x2 = int(round((x2-x0)/(scale*16)))
                y2 = int(round((y2-y0)/(scale*16)))
                heatmap[y1:y2,x1:x2] += val
                
            if doboxes==True:
                binmap = heatmap.normalize(inplace=False)
                binmap.threshold(mini=0.2, inplace=True)
                binmap = np.squeeze(binmap.view(np.ndarray))
                nb_label = label(binmap, output=binmap)
                draw_boxes = []
                for n,_ in enumerate(range(nb_label),1):
                    area = (binmap==n)
                    sumy = np.sum(area, axis=1)
                    y1 = np.argmax(sumy>0)
                    if np.all(sumy[y1:]>0):     y2 = len(sumy)
                    else:                       y2 = y1 + np.argmin(sumy[y1:]>0)
                    x1 = np.argmax(area[(y1+y2)//2])
                    x2 = np.mean(sumy[y1:y2])+x1
                    x1 = x0 + int(round(x1*scale*16))
                    y1 = y0 + int(round(y1*scale*16))
                    x2 = x0 + int(round(x2*scale*16))
                    y2 = y0 + int(round(y2*scale*16))

                    if not(blockmap is None):
                        # Eliminate ghosts of boxes which are now detected at a larger scale
                        # Those areas are still present in the heatmap of the smaller scale
                        # Ideally it should not be a cumulative blockmap, but just the next higher scale.
                        intersection = np.sum(blockmap[y1:y2, x1:x2])
                        if intersection == (y2-y1)*(x2-x1):  continue
                    draw_boxes.append(((x1,y1),(x2,y2)))
                return draw_boxes
            return None

        def seen_car(car, x, y, w, h, scale):
            """
            Update a vehicle's record when it is seen
            """
            if car['seen'] != False:
                # Car has already been detected at a larger scale in the same frame: replace
                car['recentpos'].pop()
            else:
                if len(car['recentpos'])==car['recentpos'].maxlen: car['recentpos'].popleft()
                car['nb_detection'] += 1
                car['total_detect'] += 1
            car['recentpos'].append((x,y,w,h))
            car['xc'] = int(np.mean([ pos[0] for pos in car['recentpos'] ]))
            car['scale'] = scale
            car['nb_nondetection'] = 0
            car['seen'] = scale            # Reset to False at the beginning of every frame
            return

        def new_car(x, y, w, h, scale):
            car = dict()
            car['recentpos'] = deque([],10)
            car['nb_detection'] = 0
            car['total_detect'] = 0
            car['seen'] = False
            seen_car(car, x, y, w, h, scale)  # sets 'seen' to appropriate value
            #print('New car at %d,%d, scale %4.2f'% (x,y,scale))
            return car

        # Empty record for state data
        class State(object):
            pass

        # Beginning of method
        #####################

        # Recover state information
        # Priority 1 is from self, priority 2 is from static storage
        state = getattr(self, 'find_cars_state', RoadImage.find_cars.state)
        if state is None:
            state = State()

        if not(hasattr(state,'vehicles')) or not(isinstance(state.vehicles, list)):
            print('Initialisation of RoadImage.find_cars().')
            if state is RoadImage.find_cars.state:
                print('Using static state record.')
            tunables = {
                'scales', 'scan_scales', 'areas', 'journal', 'track', 'draw_boxes',
                # feature vector configuration
                'colorspace', 'nb_orient', 'pix_per_cell', 'cell_per_block', 'spatial_size', 'hist_bins',
                # tracking algorithm
                'queue_depth', 'stop_tracking', 'new_car', 'recent',
            }
            tuned = tunables.intersection(set(dir(state)))
            for attr in tuned:
                print('   ',attr,'=',getattr(state,attr,None))
            
        _state = lambda attr, val: setattr(state,attr,getattr(state,attr,val))
        # What we maintain in state:
        # List of scales
        _state('scales', [4.,2.,1.5,1.,0.5,0.3])  # correspond to box sizes: 256,128,96,64,32,20
        _state('scan_scales',[0,1,2,3])           # index of scales scanned at every frame
        _state('areas', [((  0,336),(1280,656)), ((  0,400),(1280,592)),
                         ((200,400),(1080,544)), ((340,400),( 940,496)),
                         ((400,400),( 880,448)), ((400,400),( 880,435))] )
        _state('journal', None)                   # logging
        _state('track', None)                    # Image logging: heatmaps, blockmaps
        _state('draw_boxes', False)              # Whether to draw raw boxes from heatmaps or just tracks
        # Feature vector extraction
        _state('colorspace','YCC')
        _state('nb_orient', 9)
        _state('pix_per_cell', 8)
        _state('cell_per_block', 2)        # Each block has 4 cells
        _state('spatial_size', (32,32))    # Size of spatially sampled representation
        _state('hist_bins', 32)            # Nb of bins for histograms of colors
        # Tracking parameters: in noise, any detection will lead to 'queue_depth' subsequent detections
        _state('queue_depth', 6)           # How long old windows are kept in heat maps
        _state('stop_tracking', 2)         # Number of frames without detection to drop a track
        _state('new_car', 12)              # Total number of detections required to display a track
        _state('recent', 0)                # Max nb of non-detection to consider a track is recent
        # Classifier and scaler
        _state('clf_file', None)           # Pickle file containing the classifier and the scaler
        _state('clf', None)
        _state('scaler', None)
        # Maps
        _state('heatmaps', [])             # List of scaled heatmaps per scale
        _state('hm_scales',[])             # Scale each heatmap corresponds to
        _state('oldboxes', [])             # Hold up to ten old lists of boxes per scale (append/popleft)
        # Vehicles found
        _state('vehicles',[])              # A list of dictionaries of per-vehicle data
        # Misc
        _state('counter',0)                # Counter of calls.
        # Store state in self too
        self.find_cars_state = state       # Store in attribute
        RoadImage.find_cars.state = state  # Store in static record

        # Load classifier
        if state.clf_file != clf_file:
            import pickle
            try:
                dict_pickle = pickle.load(open('trained_svc.p','rb'))
                state.clf = dict_pickle['svm']
                state.scaler = dict_pickle['scaler']
            except (FileNotFoundError, KeyError) as e:
                raise ValueError('RoadImage.find_cars: invalid classifier file.\n%s'%str(e))
            else:
                state.clf_file = clf_file

        # Mark all vehicles as unseen
        for v in state.vehicles: v['seen'] = False
        
        # Do color conversion once for all scales
        img = self.convert_color(state.colorspace)
        width, height = img.get_size()
        
        # Check and update heatmap related lists, keeping them in sync with state.scales tunable
        hm_check = [ scale1 == scale2 for scale1, scale2 in zip(state.scales, state.hm_scales) ]
        newlen = len(state.scales)

        heatmaps = [None]*newlen
        oldboxes = [None]*newlen
        for ix in state.scan_scales:
            if ix >= len(state.areas):
                raise ValueError("RoadImage.find_cars: 'scan_scales' index %d is out of range in 'areas'."
                                 % ix)
            if ix >= len(state.scales):
                raise ValueError("RoadImage.find_cars: 'scan_scales' index %d is out of range in 'scales'."
                                 % ix)
            scale = state.scales[ix]
            x1,y1 = state.areas[ix][0]
            x2,y2 = state.areas[ix][1]
            xspan = int(round((x2-x1)/(scale*16)))
            yspan = int(round((y2-y1)/(scale*16)))
            if ix>=len(hm_check) or hm_check[ix]==False:
                # Reinitialize
                #print('Allocating heatmap for scale %4.2f size (%d,%d)' % (scale, xspan, yspan))
                heatmaps[ix] = RoadImage(np.zeros(shape=(yspan,xspan,1), dtype=np.float32),
                                         src_cspace='GRAY')
                heatmaps[ix].binary = False
                oldboxes[ix] = deque([],state.queue_depth) 
            else:
                # Reuse
                heatmaps[ix] = state.heatmaps[ix]
                oldboxes[ix] = state.oldboxes[ix]
                
        state.heatmaps = heatmaps
        state.oldboxes = oldboxes
        state.hm_scales = state.scales

        cumulative_bm = np.zeros_like(img[0].channel(0), dtype=np.bool)
        blockmap = cumulative_bm  # Careful: becomes distinct when it is initialized in the loop
        
        drw_boxes = []
        
        # Scan scales scanned at every frame
        for ix in state.scan_scales:
            scale = state.scales[ix]

            bb = find_at_scale(img[0], state.areas[ix], scale, cumulative_bm, state.clf, state.scaler,
                               state.colorspace, state.nb_orient, state.pix_per_cell,
                               state.cell_per_block, state.spatial_size, state.hist_bins)
            # Heat map update
            heatmap = state.heatmaps[ix]
            offset = state.areas[ix][0]   # heatmaps are scaled down and cover only the scanned area
            if len(state.oldboxes[ix]) == state.oldboxes[ix].maxlen:
                oldboxes = state.oldboxes[ix].popleft()
                apply_heat(heatmap, None, scale, offset, oldboxes, -1, doboxes=False)
            boxes = apply_heat(heatmap, blockmap, scale, offset, bb, 1)
            state.oldboxes[ix].append(bb)
            
            # Associate boxes to vehicles, taking into account merging/splitting, adding vehicles
            # Each vehicle is a dict. See seen_car() and new_car() above.
            large_boxes = []

            if state.counter == state.track:
                print('all cars: ',end='')
                for car in state.vehicles:
                    print(car['recentpos'][-1],' ', end='')
                print('')
                
            for box in boxes:
                # Compute box centre
                x1,y1 = box[0]
                x2,y2 = box[1]
                xcenter = .5*(x1+x2)
                ycenter = .5*(y1+y2)
                # extract known tracks and confirmed cars at same position
                nearby_all = [ car for car in filter(lambda v: v['xc']>x1 and v['xc']<x2 and\
                                                     (v['seen']==False or v['seen']>=scale),
                                                     state.vehicles)]
                known_cars = [ car for car in nearby_all if car['total_detect'] >= state.new_car ]
                known_tracks = [ car for car in nearby_all if car['total_detect'] < state.new_car ]
                
                if state.counter == state.track:
                    print('box: %s, cars: '% str(box), end='')
                    for car in known_cars:
                        print(car['recentpos'][-1],' ', end='')
                    print('')
                if len(known_cars)>1:
                    # Build a vector telling which cars have been seen recently
                    seen_recently = [ car['nb_nondetection']<state.recent for car in known_cars ]
                    if all(seen_recently):
                        # If all the cars have been seen recently, eliminate the large box
                        # and attempt detection at a smaller scale to discriminate the cars
                        large_boxes.append(box)
                    elif any(seen_recently) == False:
                        # None of the confirmed cars has been seen recently: check new tracks
                        if len(known_tracks)>0:
                            # Consider that we have seen all those tracks
                            # TODO: give points proportional to inverse of distance???
                            for car in known_tracks:
                                seen_car(car, xcenter, ycenter, x2-x1, y2-y1, scale)
                        elif scale >=1 :
                            # Nothing has been seen recently here: new car!
                            state.vehicles.append(new_car(xcenter,ycenter,x2-x1,y2-y1,scale))
                    else:
                        # Associate detection to car: most recently seen, closest to scale and center
                        # Choosing only one lets the others die slowly
                        distfunc = lambda car: car['nb_nondetection']*100 + \
                                   abs(car['scale']-scale)*100 + abs(car['xc']-xcenter)
                        index = np.argsort([distfunc(car) for car in known_cars])
                        car = known_cars[index[0]]
                        seen_car(car,xcenter,ycenter,x2-x1,y2-y1,scale)
                elif len(known_cars)==1:
                    # Associate detection to that car
                    seen_car(known_cars[0],xcenter,ycenter,x2-x1,y2-y1,scale)
                elif len(known_tracks)>0:
                    # Matches unconfirmed tracks: tag all of them
                    for car in known_tracks:
                        seen_car(car, xcenter, ycenter, x2-x1, y2-y1, scale)
                elif scale >= 1:
                    # New unconfirmed track! Don't initiate tracks for small scales: too much noise.
                    state.vehicles.append(new_car(xcenter,ycenter,x2-x1,y2-y1,scale))
                    
            for box in large_boxes: boxes.remove(box)
            
            # Search at smaller scales for 'lost' vehicles
            # Smoothing of box locations
            # Make single list of boxes for all scales
            drw_boxes.extend(boxes)
            # Update blockmap: prevent search at smaller scale in the same areas
            blockmap = np.zeros_like(img[0].channel(0), dtype=np.bool)
            for box in boxes:
                x1,y1 = box[0]
                x2,y2 = box[1]
                blockmap[y1:y2,x1:x2] = True
                cumulative_bm[y1:y2,x1:x2] = True
            blockmap.track(state)
            cumulative_bm.track(state)
                
        # Draw boxes
        del img # might cause self to be read-only

        if state.draw_boxes == True:
            color = [0,0,255]
            thick = 2
            for bb in drw_boxes:
                # draw each bounding box on your image copy using cv2.rectangle()
                cv2.rectangle(self[0], bb[0], bb[1], color, thick)

        # Draw and decay cars
        color = [255,0,0]
        thick = 6
        cars_gone = []
        for car in state.vehicles:
            if car['seen'] == False:
                car['nb_detection'] = 0
                car['nb_nondetection'] += 1
                if car['nb_nondetection'] >= state.stop_tracking:
                    cars_gone.append(car)
            elif car['total_detect'] == state.new_car:
                x,y,_,_ = car['recentpos'][-1]
                carscale = car['scale'] 
                print('New car at %d,%d, scale %4.2f'% (x,y,carscale))
            if car['total_detect'] >= state.new_car:
                #import pdb; pdb.set_trace()
                recentpos = np.array([ list(pos) for pos in car['recentpos'] ])
                pos = np.mean(recentpos, axis=0).astype(np.int)
                cv2.rectangle(self[0], (pos[0]-pos[2]//2,pos[1]-pos[3]//2),
                                       (pos[0]+pos[2]//2,pos[1]+pos[3]//2), color, thick)

        for car in cars_gone:
            xcenter,ycenter,_,_ = car['recentpos'][-1]
            carscale = car['scale']
            tot_seen = car['total_detect']
            if tot_seen >= state.new_car:
                print('Car gone at %d,%d, scale %4.2f, seen %d times'% (xcenter,ycenter,carscale,tot_seen))
            state.vehicles.remove(car)
            
        # Visualize heat map: make a global heat map
        if vis_heatmap == True or state.track != None:
            global_hm = np.zeros_like(self.channel(0),dtype=np.float32)
            for ix in state.scan_scales:
                scale = state.scales[ix]
                heatmap = state.heatmaps[ix]
                w,h = heatmap.get_size()
                heatmap = heatmap.resize(w=int(w*scale*16), h=int(h*scale*16))
                heatmap.track(state)
                w,h = heatmap.get_size()
                x1,y1 = state.areas[ix][0]
                global_hm[0,y1:y1+h,x1:x1+w]+=heatmap

            global_hm.track(state)

        state.counter += 1
        
        if vis_heatmap == True:
            return self, global_hm
        
        return self
    
    
    # List of operations which update automatically when the parent RoadImage is modified.
    # Currently this is only supported for operations implemented as numpy views.
    AUTOUPDATE = [ 'flatten', 'crop', 'channels', 'ravel', '__slice__' ]
    
