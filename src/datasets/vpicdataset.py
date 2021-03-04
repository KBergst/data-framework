""" Holds the VPIC dataset handling class."""
import numpy as np
import scipy.integrate as integ
import pyvpic
from dataframework.src.datasets.dataset import Dataset


class VPICDataset(Dataset):
    """ Abstract base class for representations of data from VPIC
    simulations

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
        NOTE! DEFAULT LABEL FOR EACH SPATIAL DIMENSION GOES Z,Y,X!!!!
        so for 2d simulation with degenerate y, would have [zcoords,xcoords]

    Methods
    -------
    ndslice(timelims = None, set_pts = None, interp = 'linear')
        returns a subsection (slice) of the data
    export(filename = 'dset.out')
        saves the Dataset object to a file

    """

    def __init__(self, vpicfiles=None, _datapkg=None, dsfile=None, **kwargs):
        """ Initializes the VPICDataset object from file or from a list

        Parameters
        ----------
        vpicfiles : str or 2-element list, default None
            File path(s) of a vpicfile(s) (anything usable by pyvpic) to make
            the Dataset out of. If using h5, need to include global.vpc
            for the parameters
        datapkg : list, optional
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
        """

        super().__init__(_datapkg=_datapkg, dsfile=dsfile, **kwargs)

        # If reading from VPIC output use special initialization
        if vpicfiles is not None:
            self._init_vpicfile(vpicfiles, **kwargs)

    def _init_vpicfile(self, vpicfiles, interleave=False, get_vars=list('all'),
                       **kwargs):
        """ Initializes the VPICDataset from VPIC output files

        Paramters
        ---------
        vpicfiles : 2-element list
            File paths of  vpicfiles (anything usable by pyvpic) to make
            the Dataset out of. Structure is [data_file, param_file]
            where data_file is for example 'data.h5' or 'global.vpc',
            param_file is 'info'
        interleave : bool, default False
            Specifies whether the associated vpic data is interleaved or not.
            Required for the data to be read in correctly.
        get_vars : list, default ['all']
            Allows the user to specify which variables they want to put
            into the Dataset. Default ['all'] takes all
        """

        self._get_params(vpicfiles[1], **kwargs)
        # get vpic data
        reader = pyvpic.open(vpicfiles[0], interleave=interleave, **kwargs)
        raw_vars = reader.datasets
        full_mesh = [0, 0, 0]  # z,y,x (CHECK THIS IS OK)
        t_dset, *full_mesh = reader.get_grid(raw_vars[0])
        self.timeseries = t_dset  # default simulation timeline
        empty_dims = []
        for i in range(3):  # default mesh takes only non-redundant dimensions
            if len(full_mesh[i]) > 1:
                self.default_mesh.append(full_mesh[i])
            else:
                empty_dims.append(i)

        if get_vars[0] != 'all':  # edit raw_vars to only have these vars /
            new_raw_vars = []     # there has to be a more efficient way...
            for get_var in get_vars:
                for raw_var in raw_vars:
                    if get_var in raw_var:  # checking if contained substring /
                        new_raw_vars.append(raw_var)  # won't handle edge /
                        break  # cases like if get_var is 'fields', case sens.
            raw_vars = new_raw_vars  # update raw_vars

        for var in raw_vars:
            var_trimmed = var.split('/')[-1]
            full_var_mesh = [0, 0, 0]  # placeholder
            var_t, *full_var_mesh = reader.get_grid(var)
            if all(var_t == self.timeseries):  # share data in memory (?)
                var_t = self.timeseries
            var_mesh = []
            if all(list(np.all(full_mesh[i] == full_var_mesh[i])
                   for i in range(len(full_mesh)))):  # maybe python can /
                var_mesh = self.default_mesh  # do this already, but idk
            else:
                for i in range(3):
                    if i not in empty_dims:
                        var_mesh.append(full_var_mesh[i])
            data = np.squeeze(reader[var][:])  # delete all len(1) dims
            self._add_var(var_trimmed, var_t, var_mesh, data)

    def _get_params(self, paramfile, paramlist=None, **kwargs):
        """ Gets parameters from an input file.

        Parameters
        ----------
        paramfile : str
            location of the file which contains the parameter information.
        paramlist : list, default None
            list of strings of the desired parameters (must follow paramfile
            syntax) if None, takes all parameters
        """
        # TODO: implement this
        print("NO PARAMS ADDED, FUNCTIONALIITY NOT ADDED YET!!!! SORRY")

    def calc_fluxfn(self, b1_name='bx', b2_name='bz',
                    cumul_integrator=integ.cumtrapz, **kwargs):
        """
        Calculates the flux function for 2d magnetic field data.

        ASSUMES the magnetic field vectors are on the same mesh

        Parameters
        ----------
        bl_name : str, default 'bx'
            names variable to be used as the magnetic field component
            in the first direction
        b2_name : str, default 'bz'
            names variable to be used as the magnetic field component
            in the second direction
        cumul_integrator : function f(y, x, **kwargs)
            function that will do the cumulative integration.
            Should return array of length one less than the length of the
            axis integrated along (like scipy.integrate.cumtrapz
            with initial = None) and (at least have the option) to integrate
            along the last axis
        ** kwargs : dict
            any additional keyword arguments the cumulative integrator needs
        """
        b1 = self.variables[b1_name]
        b2 = self.variables[b2_name]

        if len(self.default_mesh) != 2:
            ValueError("Flux function can only be calculated on"
                       "2-dimensional meshes, dataset is"
                       f"{len(self.default_mesh)}-dimensional")
        elif not (np.array_equal(b1.mesh[0], b2.mesh[0]) and
                  np.array_equal(b1.mesh[1], b2.mesh[1])):
            ValueError(f"Given magnetic field components {b1_name} and"
                       f"{b2_name} do not have the same mesh, so this"
                       " flux function calculating method is not supported."
                       " You'll need to make something up yourself.")
        else:
            flux_fn = np.zeros_like(b1.data)  # flux_fn[:,0,0] = 0
            # integrate the initial value along the first space dimension
            flux_fn[:, 1:, 0] = cumul_integrator(b2.data[:, :, 0],
                                                 b2.mesh[0], axis=1)
            # integrate everything along the second space dimension
            flux_fn[:, :, 1:] = cumul_integrator(b1.data, b1.mesh[1],
                                                 axis=2)
        # add the new variable
        self._add_var('flux_fn', b1.timeseries, b1.mesh, flux_fn)
