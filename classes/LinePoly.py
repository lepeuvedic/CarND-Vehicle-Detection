import numpy as np
from .Line import Line

class LinePoly(Line):

    def __init__(self, order=4):
        # Add an attribute for polynomial order
        super().__init__()
        self._default_order = order
        # Add def of y=1 under tag 'one' (mandatory)
        pone = [1] + [0]*order
        self._geom[('one',)] = { 'poly':pone }
        return None

    def fit(self, key, x, y, *, order=None, wfunc=None, deltay=None):
        """
        Fit a geometry controlled by additional arguments **kwargs to the points
        given by coordinate arrays x and y, assuming y=f(x).
        This implementation allows a maximum order and per point weights.
        If key is a key, delegate to super() the data conditioning and call back
        (func argument must be pointing to a partial of this method).
        """
        from functools import partial

        if order is None: order = self._default_order
        if order > self._default_order:
            raise ValueError('LinePoly.fit: You must increase default order to match requested order %d.' % order)
        
        if type(key) is tuple and key in self._geom:
            if deltay:
                raise ValueError('LinePoly.fit: deltay is reserved for library use only.')
            return super().fit(key, x, y, func=partial(self.fit,order=order,wfunc=wfunc))

        # Compute the weights (maybe should check that the weights are reasonably large?)
        if wfunc: w = wfunc(x,deltay)
        else:     w = None
        
        if len(x)<=order:
            raise ValueError('RoadImage.curves: Not enough points to fit. Is the image black?')
        
        # Fit (returns the polynomial coefficients and the covariance matrix)
        self._geom[key] = { 'op':'fit' } 
        if len(x)>order+10:
            p, V = np.polyfit(x, y, order, w=w, cov=True)
            # Store result
            self._geom[key]['cov'] = V
        else:
            p = np.polyfit(x, y, order, w=w, cov=False)
        # Extend with zeros to default order and reverse order
        q = np.zeros(self._default_order+1)
        q[:len(p)] = p[::-1]
        # Store result
        self._geom[key]['poly']=q
        
    def eval(self, key, *, z, der=0):
        """
        Computes real world x coordinates associated to z coordinates supplied as a numpy array.
        z coordinates are distances from the camera.
        key can be either a tuple or a polynomial, which is used directly
        """
        if issubclass(type(key),tuple):
            P = self._geom[key]['poly'].copy()
        else:
            P = key.copy()
        # compute derivatives of polynomial
        while der>0:
            P = [ n*p_ for n,p_ in enumerate(P[1:],1) ]
            der -= 1
            
        # one line polynomial evaluation independent of P order: returned type is same as z
        return sum( p_ * z**n for n,p_ in enumerate(P))

    def tangent(self, key, *, z, order=1):
        """
        Simplifies key to a geomtry of order=order, which is tangent to key at z=z.
        """
        Qk = self._geom[key]['poly']
        Qk = np.trim_zeros(Qk,'b')
        if len(Qk)==0: return   # Qk is already of very low order since Qk=0
        k = len(Qk)-1   # order of polynom Qk
        if k<= order: return
        
        from math import factorial as fact

        def coef(i,j,z):
            if i<j: return 0
            return fact(i) * z**(i-j) / fact(i-j)

        def mat(n,k,x0):
            return [[coef(i,j,x0) for i in range(k+1)] for j in range(n+1)]

        # Eliminate trailing zeros otherwise matrix A is singular (when order of P1 is less than len(P1)-1)
        B = mat(order, k, z)  # B is an 0:n-by-0:k matrix
        b = np.dot(B,Qk)
        A = mat(order, order, z)
        a = np.linalg.solve(A,b)

        # Extend with zeros to default order
        Pn = np.zeros(self._default_order+1, dtype=np.float)
        Pn[:len(a)] = a
        # Store result
        self._geom[key]['poly']=Pn
        self._geom[key]['order']=order
        return
        
    def delta(self, key1, key2):
        """
        Assumes that key1 and key2 describe the same geometry with an offset in the origin.
        Returns an estimate of that offset which can be given to 'move' as argument origin.
        """
        # For polynoms X = P(Y), the X offset is P2[0]-P1[0] and the Y offset is the other formula.
        # May not be robust, except on geometries made by 'move'.
        # It is impossible to distinguish X motion from Y motion on order 1 (straight lines),
        # unless one happens to know the expected distance.
        def coef(P,i,j):
            from math import factorial as fact
            if i>=len(P) or j>i: return 0
            return P[i] * (fact(i)//fact(j)//fact(i-j))
        
        P1 = self._geom[key1]['poly']
        P2 = self._geom[key2]['poly']
        
        # Eliminate trailing zeros otherwise matrix A is singular (when order of P1 is less than len(P1)-1)
        p1 = np.trim_zeros(P1,'b')
        p2 = np.trim_zeros(P2,'b')
        
        # Exchange p1, p2 to use highest order as p1.
        if len(p2)>len(p1):  p1, p2, exchange = p2, p1, True
        else:                exchange=False
            
        if len(p1)>1:
            p2 = p2[:len(p1)]    # Cut P2 at same length, because they must be the same order if P2=P1.move(X,Y)
            A = np.array([ [ coef(p1,i,j) for i,_ in enumerate(p1,j) ] for j,_ in enumerate(p1,0) ])
            powers = np.linalg.solve(A,p2)

            # Theory does not work for order 1 or order 0 polynoms, which can happen (straight lines).
            # It is not possible to assess speed based on relative motion w.r.t. a line: unless
            # the line is exactly perpendicular to motion or we have another info, there is no
            # way to distinguish X motion from Y motion. When order is 1, we assume X motion is zero.
            if len(powers)>=3:
                Y = powers[:-1] # last component contains info about X. Y = [y**0, y**1,...]
            else:
                # Assume x motion is zero, evaluate y using all three values in powers
                Y = powers
                x = 0
            lny = np.log(abs(Y))
            l = np.polyfit(np.arange(len(lny)),lny,1,cov=False)
            print('              zero offset=',l[1])
            y = float(np.exp(l[0]))
            if len(powers)>=3:
                powers[-1] = y**(len(powers)-1)
                x = np.dot(A[0],powers)-p2[0]
        else:
            # len(p1) and len(powers) = 1, A is a scalar.
            # P1 and P2 are mere constants. Assume pure lateral motion
            x=P2[0]-P1[0]
            y=0

        if exchange:  x,y = -x,-y
        return (x,y)

    def move(self, key, *, origin, dir, key2=None):
        """
        origin is a tuple, a vector (x,y) from the current axes'origin and the new origin.
        The vector should be estimated based on car speed and direction.
        dir is a unit length vector giving the new "ahead" direction.
        The geometry associated to key is represented in the new axes and
        associated with key2 too if supplied.
        """
        if dir!=0:
            raise NotImplementedError('LinePoly.move: direction change is not implemented.')
        X,Y = origin
        P1=self._geom[key]['poly']

        def Comb(n,p):
            from math import factorial as fact

            return fact(n)//fact(p)//fact(n-p)
        
        P2 = P1.copy()
        P2 = np.array([ np.sum( ai * Comb(i,j) * Y**(i-j) for i,ai in enumerate(P1[j:],j) ) for j,_ in enumerate(P1,0) ])
        P2[0] -= X
        self._geom[key]['poly'] = P2
        # Call parent to perform key management
        super(LinePoly,self).move(key, origin=0, dir=0, key2=key2)
        
    def blend(self, key, *, key1, key2, op, **kwargs):
        """
        Blends two line definitions to make a new one.
        Two operations are currently supported: wsum and wavg
        'wavg', w2=       --> g = (1-w2) g1 + w2 g2
        'wavg', w1=, w2=  --> g = (w1 g1 + w2 g2)/(w1+w2)    w1+w2 != 0
        'wsum', w1=, w2=  --> g = w1 g1 + w2 g2
        """
        if op=='wavg' or op=='wsum':
            try:
                w2 = kwargs['w2']
            except KeyError:
                raise NameError('LinePoly.blend: arg w2 is required when op=%s.' % repr(op))
            try:
                w1 = kwargs['w1']
                if op=='wavg':
                    total = w1+w2
                    w1 /= total
                    w2 /= total
            except KeyError:
                if op=='wsum':
                    raise NameError("LinePoly.blend: arg w1 is required when op='wsum'")
                w1 = 1.-w2
            except ZeroDivisionError:
                print('LinePoly.blend: %s operator does not work when w1+w2=0'% repr(op))
                raise
            # Weighted average of polynoms is easy because they are linear
            P1 = self._geom[key1]['poly']
            P2 = self._geom[key2]['poly']
            Pout = [ p1*w1+p2*w2 for p1,p2 in zip(P1,P2) ]

            from classes import try_apply
            # Passing just min as first arg does not work when only one zmax is present. 
            z_max = try_apply(lambda *x: min(x), 0, lambda x: self._geom[x]['zmax'], KeyError, key1, key2)
        else:
            raise NotImplementedError('LinePoly.blend: operation %s is not yet implemented.' % str(op))
        # Save result
        self._geom[key] = { 'poly':Pout, 'op':op, 'w1':w1, 'w2':w2 }
        if z_max: self._geom[key]['zmax'] = z_max
        return
    
Line.Register(LinePoly, 'poly3')
