from src.variables.dynmeshvar import DynMeshVar
from src.variables.statmeshvar import StatMeshVar

class Dataset(ABC):
    """Abstract base class for representations of data from simulations and in-situ

    Attributes
    ----------    
    params : dict
        defines any parameters of the Dataset
    variables : list
        holds all the Variables of the Dataset

    Methods
    -------
    ndslice(timelims = None, set_pts = None, interp = 'linear')
        returns a subsection (slice) of the data
    export(filename = 'dset.out')
        saves the Dataset object to a file

    """

    def __init__(self, _datapkg=None, dsfile=None):
        """ Initializes the Dataset object from file or from a list

        Parameters
        ----------
        _datapkg : list, optional
            List of data used when slicing from another Dataset object
            _datapkg[0] : array_like
                the timeseries
            _datapkg[1] : array_like
                the default mesh of the data (None for in-situ data)
            _datapkg[2] : dict
                dictionary containing the parameters of the dataset
            _datapkg[3] : list
                list containing the Variable objects for the new dataset

        dsfile- str 
            filename of a dataset-exported file to reconstruct the Dataset of
        """

        self.params = {}
        self.variables = []
        self.timeseries = []
        self.default_mesh = []

        if (dsfile is not None): # init from file takes precedence
            self._init_file(dsfile)
        elif (_datapkg is not None):
            self._init_datapkg(_datapkg)
            
    def _init_file(self,dsfile):
        """ Initializes the Dataset object from a file"""
        print('initialization from file is not yet supported')
        pass
        #TODO: decide on a file format for exporting (probably hdf5?)
        #TODO: implement this

    def _init_datapkg(self,_datapkg):
        """ Initializes the Dataset object from a list

        Parameters
        ----------
        _datapkg : list, optional
            List of data used when slicing from another Dataset object
            _datapkg[0] : array_like
                the timeseries
            _datapkg[1] : array_like
                the default mesh of the data (None for in-situ data)
            _datapkg[2] : dict
                dictionary containing the parameters of the dataset
            _datapkg[3] : dict
                dictionary containing the Variable objects for the new dataset
        """
            
        self.timeseries = _datapkg[0]
        self.default_mesh = _datapkg[1]
        self.params = dict(_datapkg[2]) #shallow copy
        self.variables = list(_datapkg[3]) #shallow copy
        
    def _add_param(self, key, value, verbose = True):
        """ adds or changes a parameter of the Dataset """
        self.params[key] = value
        if verbose:
            print(f'parameter {key} = {value}')

    def _add_var(self, label, var_tseries, var_mesh, data, verbose = True):
        """ adds or changes a Variable of the Dataset """
        #TODO: adapt to account for possible different timeseries
        if (var_mesh[0].hasattr('__len__')): #not a static 1d mesh
            if len(var_mesh[0]) != len(var_mesh[1]): #not a static nd mesh
                self.variables.append(DynMeshVar(label, var_tseries, var_mesh, data))
            else: # a static nd mesh
                self.variables.append(StatMeshVar(label, var_tseries, var_mesh, data))
        else: #a static 1d mesh
            self.variables.append(StatMeshVar(label, var_tseries, var_mesh, data))
                
        if verbose:
            print(f'Added {label} Variable')

    def export(self, filename = 'dset.out'):
        """ Saves the Dataset object information to a file

        Parameters
        ----------
        filename: string, default 'dset.out'
           string declaring where the Dataset file path
        """

    def ndslice(self, timelims = None, set_pts = None, interp = 'linear'):
        """ Returns a Dataset which is a slice of the current Dataset.

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
        slicedset : Dataset
            the slice of the Dataset
        """

        new_vars = []

        for var in self.variables:
            new_vars.append(var.ndslice(timelims = timelims, zooms = zooms,
                                        set_pts = set_pts, interp = interp)
        

        
