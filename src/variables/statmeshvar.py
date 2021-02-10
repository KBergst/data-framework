from variable import Variable
import numpy as np
import scipy.interpolate as interp
#current implementation is variable data, timeseries, mesh are 
# all numpy arrays. data is indexed [time,spacedim1,spacedim2,...]
#TODO: expand functionality for mesh to be made of meshgrids?

def StatMeshVar(Variable):
    """ Class for representations of specific measured values
        from simulations, which have an unchanging measurement point.

    Attributes
    ----------
    label : string
        labels the type of data contained in the Variable
    timeseries : array
        the timeseries on which the data was taken
    mesh : list of arrays
        the time-independent mesh where the data was taken
        [arr(z-coords), arr(y-coords), arr(x-coords)]
    data : array
        the variable's data, indexed data[time_idx][space_idxs]

    Methods
    -------
    ndslice() : StatMeshVar
        return a new StatMeshVar which is a subsection of this StatMeshVar
    """

    def ndslice(self, timelims = None, zooms = None, set_pts = None, 
                interp = 'linear'):
        """ Returns a Variable which as a slice of the current Variable
        
        Parameters
        ----------
        timelims : list, default None
            list containing [tmin,tmax] for the slice. 'None' takes the whole timeseries.
        zooms : array (N,2) default None
            array specifying boundaries for zooming into the data
            use +/- np.inf if no zoom along that axis (as one would expect)
        set_pts : list, default None
            2 or 3-element list containing spatial coordinates defining the nd section
            of the data to take. 'None' takes the whole space (should be used for in-situ data).
        interp : string, default 'linear'
            determines interpolation for values in between the mesh
            
        Returns
        -------
        slicedvar : Variable
            the slice of the Variable

        """

        super().ndslice(timelims = timelims, zooms = zooms, 
                        set_pts = set_pts, interp = interp)

    def _timeslice(self, timelims):
        idxs = np.logical_and( self.timeseries >= timelims[0],
                               self.timeseries <= timelims[1])
        #slice the data and the timeseries
        self.timeseries = self.timeseries[idxs]
        self.data = self.data[idxs,:]

    def _zoom(self, zooms):
        for dim in range(len(self.mesh)):
            idxs = np.logical_and( self.mesh[dim] >= zooms[dim][0],
                                   self.mesh[dim] <= zooms[dim][1])
            #slice the mesh and the data for each dimension
            self.mesh[dim] = self.mesh[dim][idxs]
            self.data = np.compress(idxs, self.data, axis = dim+1) #ax 0=t

    def _spaceslice(self, set_pts, interp):
        #can accomodate multiple interpolation options if we need to 
        #right now just going to do linear ND
        pts = np.stack(self.mesh).transpose
        if interp == 'linear':
            datainterp = interp.LinearNDInterpolator(pts,self.data)

        else:
            raise ValueError(r"Specified interpolation type {interp} "+
                             "is not currently implemented")
