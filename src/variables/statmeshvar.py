""" class to represent a variable which is defined on an unchanging
    rectangular mesh. """
from dataframework.src.variables.variable import Variable
import numpy as np
import scipy.interpolate
# current implementation is variable data, timeseries, mesh are
# all numpy arrays. data is indexed [time,spacedim1,spacedim2,...]
# TODO: allow for a spacetime-slice which is e.g. a spacecraft path
#    this is different from ndslice since ndslice only supports zooming in
#    in time, and spacecraft-like datasets haven't been implemented yet.


class StatMeshVar(Variable):
    """ Class for representations of specific measured scalar values
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

    def bounds(self, time=True, space=True):
        """Returns the limits of the mesh and/or timeseries

        Parameters
        ----------
        time: bool, default True
            determines if time bounds are returned
        space: bool, default True
            determines if space bounds are returned

        Returns
        -------
        bounds: list of lists
            returns [[min0, max0], ... [minN, maxN]] for dimensions 0-N
            if time is returned it is the zeroth dimension
        """
        bounds = []

        if time:
            bounds.append([min(self.timeseries), max(self.timeseries)])
        if space:
            for i in range(len(self.mesh)):
                bounds.append([min(self.mesh[i]), max(self.mesh[i])])
        bounds = np.array(bounds)
        return bounds

    def ndslice(self, timelims=None, zooms=None, set_pts=None,
                interp='linear', **kwargs):
        """ Returns a Variable which as a slice of the current Variable
        
        Parameters
        ----------
        timelims : list, default None
            list containing [tmin,tmax] for the slice. 'None' takes the whole
            timeseries. Slices must be static in time.
        zooms : array (N,2) default None
            array specifying boundaries for zooming into the data
            use +/- np.inf if no zoom along that axis (as one would expect)
        set_pts : list of arrays, default None
            2 or 3-element list containing spatial coordinates defining
            the nd section of the data to take. 'None' takes the whole space
            (should be used for in-situ data).
        interp : string, default 'linear'
            determines interpolation for values in between the mesh
        ** kwargs : dict
            holds keyword arguments (if any) passed from the 
            encompassing dataset (if any)
            
        Returns
        -------
        slicedvar : Variable
            the slice of the Variable

        """

        sliced_var = super().ndslice(timelims=timelims, zooms=zooms,
                                     set_pts=set_pts, interp=interp, **kwargs)

        return sliced_var

    def _timeslice(self, timelims, **kwargs):
        idxs = np.logical_and(self.timeseries >= timelims[0],
                              self.timeseries <= timelims[1])
        # slice the data and the timeseries
        self.timeseries = self.timeseries[idxs]
        self.data = self.data[idxs, :]

    def _zoom(self, zooms, **kwargs):
        for dim in range(len(self.mesh)):
            min_max = np.sort(zooms[dim])
            idxs = np.logical_and(self.mesh[dim] >= min_max[0],
                                  self.mesh[dim] <= min_max[1])
            # slice the mesh and the data for each dimension
            self.mesh[dim] = self.mesh[dim][idxs]
            self.data = np.compress(idxs, self.data, axis=dim+1)  # ax 0=t

    def _spaceslice(self, set_pts, interp, **kwargs):
        # can accomodate multiple interpolation options if we need to
        # right now just going to do linear ND ON A GRID
        # only 1D SLICE (line intersect) is supported right now!!!
        # TODO: make this less of a mess (premade stuff?),
        #     maybe move some stuff to utilities
        # have to flatten the array in the space dimensions to get \
        # it to interpolate on the whole mesh

        # check for interpolation type override from dataset default
        if self.label+'interp' in kwargs.keys():
            print(f"overriding interpolation type {interp} in"
                  f" variable {self.label} with {kwargs[self.label+'interp']}")
            var_interp = kwargs[self.label+'interp']
        else:
            var_interp = interp
        # stick the time dimension to the back to do the interp
        data_rearr = np.stack(tuple(self.data[i] for i in
                                    range(self.data.shape[0])), axis=-1)
        if var_interp in ('linear', 'nearest'):
            boxsize = np.fromiter((len(self.mesh[i]) for i in
                                  range(len(self.mesh))), int)
            deg_dims = boxsize <= 1  # notes dimensions with lengths 1 or 0
            if np.sum(deg_dims) < 1:
                datainterp = scipy.interpolate.RegularGridInterpolator(
                    tuple(self.mesh), data_rearr, method=var_interp)
            else:
                raise ValueError('This interpolation method cannot handle'
                                 'degenerate dimensions, please reprocesss'
                                 'to avoid dimensions with size 1')
        else:
            raise ValueError(f'Specified interpolation type {interp} ' +
                             'is not currently implemented')
        dim_mesh = len(self.mesh)
        dim_slice = len(set_pts)-1  # assumes pts not degenerate
        if not (0 < dim_slice < dim_mesh):  # python chains comparisons
            raise ValueError(f'for {dim_mesh}-dimensional mesh ' +
                             'number of set points {dim_slice+1} does not ' +
                             'define an acceptable subspace of dimension ' +
                             'less than {dim_mesh}.')
        if dim_slice == 1:
            # find unit vec along line
            # use the grid spacing of the dimension with which
            # the slice is most aligned
            # use vector eqn of line to find limits on the parameter s
            # point = point_0 +s*(unit_vec)
            vec = set_pts[1] - set_pts[0]
            unit_vec = vec/np.linalg.norm(vec)
            base_dir = np.argmax(unit_vec)  # 0, 1, 2 etc.
            base_dir_dx = (self.mesh[base_dir][1] -  # assumes const dx
                           self.mesh[base_dir][0])

            min_s = -np.inf
            max_s = np.inf

            for dim in range(len(self.mesh)):  # find endpoints in t
                direction = np.sign(unit_vec[dim])
                if direction == 0:  # no div by 0
                    continue
                if direction == 1:  # moving  positively in this dimension
                    min_dim = (self.mesh[dim][0]
                               - set_pts[0][dim])/unit_vec[dim]
                    max_dim = (self.mesh[dim][-1]
                               - set_pts[0][dim])/unit_vec[dim]
                else:  # moving negatively in this dimension
                    min_dim = (self.mesh[dim][-1]
                               - set_pts[0][dim])/unit_vec[dim]
                    max_dim = (self.mesh[dim][0]
                               - set_pts[0][dim])/unit_vec[dim]

                # refine where the line collides with the edge of the box
                min_s = max(min_s, min_dim)
                max_s = min(max_s, max_dim)
            min_s += base_dir_dx/3  # try to avoid nans maybe
            max_s -= base_dir_dx/3
            # calculate new mesh, put in list of arrays (length 1)
            mesh = [np.arange(min_s, max_s, base_dir_dx)]
            # build array in shape (timeseries,mesh)
            dat_slice = np.vstack(tuple(datainterp(set_pts[0] + t*unit_vec)
                                        for t in mesh[0])).T

        else:
            raise ValueError('slices in more than 1d are not currently ' +
                             'supported. If using the zoom method to specify' +
                             'a 2d slice parallel to one of the mesh faces ' +
                             'works for you, do that instead.')

        # set new values of variable
        self.mesh = mesh
        self.data = dat_slice
