""" Holds the abstract base class Variable """
import abc  # abstract base class handling
import copy


class Variable(abc.ABC):
    """ Abstract base class for representations of specific measured
    values from simulations and in-situ

    Attributes
    ----------
    label : string
        labels the type of data contained in the Variable
    timeseries : array
        the timeseries on which the data was taken
    mesh : list of [arrays or list of arrays]
        the mesh (time-independent or not) where the data was taken
    data : array or list of arrays
        the variable's data, indexed data[time_idx][space_idxs]

    Methods
    -------
    ndslice() : Variable
        method subclasses will use to slice
    """

    def __init__(self, label, timeseries, mesh, data):
        """ Initializes the variable object """
        self.label = label
        self.timeseries = timeseries
        self.mesh = mesh
        self.data = data

    def ndslice(self, timelims=None, zooms=None, set_pts=None,
                interp='linear', **kwargs):
        """ Returns a Variable which as a slice of the current Variable

        Parameters
        ----------
        timelims : list, default None
            list containing [tmin,tmax] for the slice.
            'None' takes the whole timeseries.
        zooms : array (N,2) default None
            array specifying boundaries for zooming into the data
            if you are doing a slice which is parallel to a face of the mesh,
            e.g. is defined by a single x,y,or z value like x = 5,
            SET IT THIS WAY using [xmin,xmax] = [5-dx/2,5+dx/2]
            for dx the grid spacing in x
        set_pts : list, default None
            2 or 3-element list containing spatial coordinates
            defining the nd section of the data to take.
            'None' takes the whole space (should be used for in-situ data).
            currently ONLY FOR 1D SLICES (2 pts) because 3d slices become weird
            shapes and are too hard for right now
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

        slicedvar = copy.deepcopy(self)

        # variable does surgery on its clone. Maybe not ~pythonic~
        # maybe instead of slicedvar use as backup then do a switcheroo
        if timelims is not None:
            slicedvar._timeslice(timelims, **kwargs)
        if zooms is not None:
            slicedvar._zoom(zooms, **kwargs)
        if set_pts is not None:
            slicedvar._spaceslice(set_pts, interp, **kwargs)

        return slicedvar

    @abc.abstractmethod
    def _zoom(self, zooms, **kwargs):
        pass

    @abc.abstractmethod
    def _timeslice(self, timelims, **kwargs):
        pass

    @abc.abstractmethod
    def _spaceslice(self, set_pts, interp, **kwargs):
        pass
