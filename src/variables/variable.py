import abc #abstract base class handling
from abc import ABC
import copy

class Variable(ABC):
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
        abstract method subclasses will use to slice
    """

    def __init__(self, label, timeseries, mesh, data):
        """ Initializes the variable object """
        self.label = label
        self.timeseries = timeseries
        self.mesh = mesh
        self.data = data

    def ndslice(self, timelims = None, zooms = None, set_pts = None, 
                interp = 'linear'):
        """ Returns a Variable which as a slice of the current Variable
        
        Parameters
        ----------
        timelims : list, default None
            list containing [tmin,tmax] for the slice. 'None' takes the whole timeseries.
        zooms : array (N,2) default None
            array specifying boundaries for zooming into the data
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

        slicedvar = copy.deepcopy(self)

        if timelims is not None:
            slicedvar._timeslice(timelims)
        if zooms is not None:
            slicedvar._zoom(zooms)
        if set_pts is not None:
            slicedvar._spaceslice(set_pts, interp)

        return slicedvar

    @abstractmethod
    def _zoom(self, zooms):
        pass

    @abstractmethod
    def _timeslice(self, timelims):
        pass

    @abstractmethod
    def _spaceslice(self, set_pts, interp):
        pass


        
        
