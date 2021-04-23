""" Holds the class for generic datasets """
from dataframework.src.variables.dynmeshvar import DynMeshVar
from dataframework.src.variables.statmeshvar import StatMeshVar


class Dataset():
    """Abstract base class for representations of data from simulations and in-situ

    Attributes
    ----------
    params : dict
        defines any parameters of the Dataset
    variables : dict
        holds all the Variables of the Dataset
    timeseries = array-like
        default timeseries of the data
    default_mesh : list of arrays or list of list of arrays
        default mesh of the data. Outer length = dim of system

    Methods
    -------
    ndslice(timelims = None, set_pts = None, interp = 'linear')
        returns a subsection (slice) of the data
    export(filename = 'dset.out')
        saves the Dataset object to a file
    """

    def __init__(self, _datapkg=None, dsfile=None, **kwargs):
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
        dsfile :  str
            filename of a dataset-exported file to reconstruct the Dataset of
        kwargs : dict
            optional keyword argument. This base class doesn't do anything
            with them, but it can hold them.
        """

        self.params = {}
        self.variables = {}  # so that we can call them by name \
        self.timeseries = []  # (not happy with this solution- more elegant?)
        self.default_mesh = []

        if (dsfile is not None):   # init from file takes precedence
            self._init_file(dsfile)
        elif (_datapkg is not None):
            self._init_datapkg(_datapkg)

    def _init_file(self, dsfile):
        """ Initializes the Dataset object from a file"""
        print('initialization from file is not yet supported')
        # TODO: decide on a file format for exporting (probably hdf5?)
        # TODO: implement this

    def _init_datapkg(self, _datapkg):
        """ Initializes the Dataset object from a list

        Parameters
        ----------
        _datapkg : list, optional
            List of data used when slicing from another Dataset object
            _datapkg[0] : array_like
                the timeseries
            _datapkg[1] : list of (arrays or list of arrays)
                the default mesh of the data (None for in-situ data)
            _datapkg[2] : dict
                dictionary containing the parameters of the dataset
            _datapkg[3] : dict
                dictionary containing the Variable objects for the new dataset
        """

        self.timeseries = _datapkg[0]
        self.default_mesh = _datapkg[1]
        self.params = dict(_datapkg[2])  # shallow copy
        self.variables = dict(_datapkg[3])  # shallow copy

    def _add_param(self, key, value, verbose=True):
        """ adds or changes a parameter of the Dataset """
        self.params[key] = value
        if verbose:
            print(f'parameter {key} = {value}')

    def _add_var(self, label, var_tseries, var_mesh, data, verbose=True):
        """ adds or changes a Variable of the Dataset """
        # TODO: adapt to account for possible different timeseries
        if (hasattr(var_mesh[0][0], '__len__')):  # not a static 1d mesh
            self.variables[label] = DynMeshVar(label, var_tseries,
                                               var_mesh, data)
        else:  # a static mesh
            self.variables[label] = StatMeshVar(label, var_tseries,
                                                var_mesh, data)

        if verbose:
            print(f'Added {label} Variable')

    def export(self, filename='dset.out'):
        """ Saves the Dataset object information to a file

        Parameters
        ----------
        filename: string, default 'dset.out'
           string declaring where the Dataset file path
        """
        raise ValueError(f"method {self.export.__name__} for class" +
                         f" {type(self)} has not yet" +
                         " been implemented, sorry")

    def ndslice(self, timelims=None, zooms=None, set_pts=None,
                interp='linear'):
        """ Returns a Dataset which is a slice of the current Dataset.

        Parameters
        ----------
        timelims : list, default None
            list containing [tmin,tmax] for the slice.
            'None' takes the whole timeseries.
        zooms : array (N,2) default None
            array specifying boundaries for zooming into the data
        set_pts : list, default None

            2 or 3-element list containing spatial coordinates
            defining the nd section
            of the data to take. 'None' takes the whole space
            (should be used for in-situ data).
        interp : string, default 'linear'
            determines interpolation for values in between the mesh

        Returns
        -------
        slicedset : Dataset
            the slice of the Dataset
        """

        new_vars = {}
        defaults_var = list(self.variables.keys())[0]  # defaults to rand var

        for orig_var in self.variables.values():
            if orig_var.mesh is self.default_mesh:
                # var for defining default mesh
                defaults_var = orig_var.label
            new_vars[orig_var.label] = orig_var.ndslice(timelims=timelims,
                                                        zooms=zooms,
                                                        set_pts=set_pts,
                                                        interp=interp)
            
        new_timeseries = new_vars[defaults_var].timeseries
        new_mesh = new_vars[defaults_var].mesh
        slicedset = self.__class__(_datapkg=[new_timeseries, new_mesh,
                                             self.params, new_vars])
        return slicedset
