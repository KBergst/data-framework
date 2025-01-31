""" Holds the VPIC dataset handling class."""
import numpy as np
# import scipy.integrate as integ
import scipy.interpolate as interp
import scipy.ndimage as nd
from skimage import measure
from skimage.segmentation import flood  # for defining structures
from skimage.morphology import binary_dilation  # for closing x point
import pyvpic
from dataframework.src.datasets.dataset import Dataset

rng = np.random.default_rng(125)

# Some helper functions
# TODO chuck these somewhere else e.g. utils


def uniform(array, atol=1e-3):
    """ Checks to see if the inputted array is sufficiently evenly spaced.
    Uses numpy.diff and numpy.allclose
    """
    return np.allclose(np.diff(array), np.diff(array)[0], atol=atol)


def ccw(A, B, C):
    """ Test whether the three points are listed in a counterclockwise order,
        but ~vectorized~
    Can't handle colinear points because I'm not handling edge cases rn
    A- array, shape (n_pts,2)
    B- array, shape (n_pts,2)
    C- array, shape (n_pts,2)
    """
    return ((C[:, 1]-A[:, 1])*(B[:, 0] - A[:, 0])
            > (B[:, 1] - A[:, 1])*(C[:, 0] - A[:, 0]))


def intersect_true(A, B, C, D):
    """ Determine whether two line segments AB and CD intersect
    A- array, shape (n_pts,2)
    B- array, shape (n_pts,2)
    C- array, shape (n_pts,2) (optional n_pts = 1)
    D- array, shape (n_pts,2) (optional n_pts = 1)
    """
    cond1 = np.logical_not(ccw(A, C, D) == ccw(B, C, D))
    cond2 = np.logical_not(ccw(A, B, C) == ccw(A, B, D))
    return np.logical_and(cond1, cond2)


def line_intersect(A, B, C, D):
    """ Finds the intersection of the lines AB and CD, if it exists
    Using https://en.wikipedia.org/wiki/Line%E2%80%93line_intersect \\
    ion#Given_two_points_on_each_line_segment
    A- array, shape (n_pts,2)
    B- array, shape (n_pts,2)
    C- array, shape (n_pts,2)
    D- array, shape (n_pts,2)
    Typically N_PTS = 1
    """
    denominator = ((A[:, 0] - B[:, 0])*(C[:, 1] - D[:, 1])
                   - (A[:, 1] - B[:, 1])*(C[:, 0] - D[:, 0]))
    px = ((A[:, 0]*B[:, 1] - A[:, 1]*B[:, 0])*(C[:, 0] - D[:, 0])
          - (A[:, 0] - B[:, 0])*(C[:, 0]*D[:, 1] - C[:, 1]*D[:, 0])) \
          / denominator
    py = ((A[:, 0]*B[:, 1] - A[:, 1]*B[:, 0])*(C[:, 1] - D[:, 1])
          - (A[:, 1] - B[:, 1])*(C[:, 0]*D[:, 1] - C[:, 1]*D[:, 0])) \
          / denominator
    p = np.stack([px, py], axis=1)
    return p


def gap_fill(array):
    """ Finds points in the 2d array that could be categorized as 'gaps'
    And returns a mask for those points
    I'm sure there's a better way to do this but oh well"""
    
    # check if there is a pixel to the center left and a pixel in
    #  one of the right spaces
    opp_left = np.logical_and(np.roll(array, 1, axis=1),
                              (np.abs(np.roll(array, (-1, 1), axis=(1, 0)))
                               + np.abs(np.roll(array, (-1, -1), axis=(1, 0)))
                               + np.abs(np.roll(array, -1, axis=1))))
    # check if there is a pixel to the center right
    #  and a pixel in one of the left spaces
    opp_right = np.logical_and(np.roll(array, -1, axis=1),
                               (np.abs(np.roll(array, (1, -1), axis=(1, 0)))
                                + np.abs(np.roll(array, (1, 1), axis=(1, 0)))
                                + np.abs(np.roll(array, 1, axis=1))))
    # check if there is a pixel to the center top
    #  and a pixel in one of the bottom spaces
    opp_up = np.logical_and(np.roll(array, 1, axis=0),
                            (np.abs(np.roll(array, (-1, 1), axis=(0, 1)))
                            + np.abs(np.roll(array, (-1, -1), axis=(0, 1)))
                            + np.abs(np.roll(array, -1, axis=0))))
    # check if there is a pixel to the center bottom
    #  and a pixel in one of the top spaces
    opp_down = np.logical_and(np.roll(array, -1, axis=0),
                              (np.abs(np.roll(array, (1, 1), axis=(0, 1)))
                               + np.abs(np.roll(array, (1, -1), axis=(0, 1)))
                               + np.abs(np.roll(array, 1, axis=0))))
    # check if either of the diagonals has both pixels
    opp_diag = np.logical_or(np.logical_and(np.roll(array, (1, 1),
                                                    axis=(0, 1)),
                                            np.roll(array, (-1, -1),
                                                    axis=(0, 1))),
                             np.logical_and(np.roll(array, (1, -1),
                                                    axis=(0, 1)),
                                            np.roll(array, (-1, 1),
                                                    axis=(0, 1))))

    gap_filled_array = np.logical_or.reduce((array, opp_left, opp_right,
                                             opp_up, opp_down, opp_diag))
    # returns ones and zeros for output readability
    return gap_filled_array.astype(np.int64)


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

    def _init_vpicfile(self, vpicfiles, interleave=False, get_vars=None,
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
        get_vars : list, default None (set to ['all'])
            Allows the user to specify which variables they want to put
            into the Dataset. Default ['all'] takes all
        """
        if get_vars is None:
            get_vars = ['all']
            
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

    def calc_fluxfn(self, b1_name='bx', b2_name='bz'):
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
        ** kwargs : dict
            any additional keyword arguments the cumulative integrator needs
        """
        b1 = self.variables[b1_name]
        b2 = self.variables[b2_name]

        if len(self.default_mesh) != 2:
            raise ValueError("Flux function can only be calculated on"
                             "2-dimensional meshes, dataset is"
                             f"{len(self.default_mesh)}-dimensional")
        if not (np.array_equal(b1.mesh[0], b2.mesh[0]) and
                np.array_equal(b1.mesh[1], b2.mesh[1])):
            raise ValueError(f"Given magnetic field components {b1_name} and"
                             f"{b2_name} do not have the same mesh, so this"
                             " flux function calculating method is not"
                             " supported. You'll need to make something up"
                             " yourself.")
        if not (uniform(b1.mesh[0]) and uniform(b1.mesh[1]) and
                uniform(b2.mesh[0]) and uniform(b2.mesh[1])):
            raise ValueError("inputted variables are not on uniform mesh."
                             " This flux function calculating method is not"
                             " Supported for this case.")

        flux_fn = np.zeros_like(b1.data)  # flux_fn[:,0,0] = 0
        # integrate the initial value along the first space dimension
        d0 = b2.mesh[0][1] - b2.mesh[0][0]
        flux_fn_ic = np.cumsum(b2.data[:, :, 0]*d0, axis=1)
        # integrate everything along the second space dimension
        d1 = b2.mesh[1][1] - b2.mesh[1][0]
        flux_fn = np.cumsum(-b1.data*d1, axis=2) + \
            np.stack([flux_fn_ic for i in range(flux_fn.shape[-1])], axis=-1)
        # add the new variable
        self._add_var('flux_fn', b1.timeseries, b1.mesh, flux_fn)

    def find_structures(self, b1_name='b1', b2_name='b2',
                        smoothing=3, time_idx=0, de_tol=5,
                        cs_limit=0.5, **kwargs):
        """
        Finds the structures around the nulls of the simulation
        for 2D VPIC data
 
       Parameters
        ----------
        b1_name : str, default 'b1'
            names variable to be used as the magnetic field component
            in the first direction
        b2_name : str, default 'b2'
            names variable to be used as the magnetic field component
            in the second direction
        smoothing : int or list, default '3'
            defines sigma of gaussian filter. Acceptable inputs are
            a single int (for all spatial dimensions), a two element
            list (for each spatial dimension), or a three element list
            (if time smoothing is desired)
        time_idx : int
           which index of time to find the structures for. Default is '0'.
        de_tol : float
           the maximum closest approach in de for a contour to be considered
           to be passing 'through' an x point
        cs_limit : float between 0 and 1
           the fraction of the regional peak current which is part of the sheet
        """
        # TODO: save all quantities which are calculated over all times to
        # speed up structure finding after the first chosen index
        print("Finding structures at simulation time" +
              f" {self.timeseries[time_idx]}")
        b1 = self.variables[b1_name]
        b2 = self.variables[b2_name]

        if len(self.default_mesh) != 2:
            raise ValueError("Flux function can only be calculated on"
                             "2-dimensional meshes, dataset is"
                             f"{len(self.default_mesh)}-dimensional")
        if not len(self.timeseries) == 1:
            raise ValueError("Currently can do this function"
                             " for ONLY ONE TIME")
        # TODO: MAKE IT POSSIBLE TO DO FOR MULTIPLE TIMES AT ONCE
        if not (np.array_equal(b1.mesh[0], b2.mesh[0]) and
                np.array_equal(b1.mesh[1], b2.mesh[1])):
            raise ValueError(f"Given magnetic field components {b1_name} and"
                             f"{b2_name} do not have the same mesh, so this"
                             " flux function calculating method is not "
                             "supported. You'll need to make something"
                             " up yourself.")
        # save the number of grid pts per de as a param
        dz_per_de = 1/(b1.mesh[0][1]-b1.mesh[0][0])
        dx_per_de = 1/(b1.mesh[1][1]-b1.mesh[1][0])
        d_per_de = int((dz_per_de + dx_per_de)/2)
        self._add_param('d_per_de', d_per_de)

        # format the smoothing input (0 gives no smoothing)
        if not hasattr(smoothing, '__len__'):
            full_smoothing = [0, smoothing, smoothing]
        elif len(smoothing) == 1:
            full_smoothing = [0] + list(smoothing) + list(smoothing)
        elif len(smoothing) == 2:
            full_smoothing = [0] + list(smoothing)
        elif len(smoothing) == 3:
            full_smoothing = smoothing
        else:
            raise ValueError(f"incompatible smoothing value {smoothing}")

        # smooth the magnetic field data
        smooth_b1_data = nd.gaussian_filter(b1.data, full_smoothing)
        smooth_b2_data = nd.gaussian_filter(b2.data, full_smoothing)
        self._add_var(b1_name+'_smooth', self.variables[b1_name].timeseries,
                      self.variables[b1_name].mesh, smooth_b1_data)
        self._add_var(b2_name+'_smooth', self.variables[b2_name].timeseries,
                      self.variables[b2_name].mesh, smooth_b2_data)

        if 'flux_fn' not in self.variables.keys():
            self.calc_fluxfn(b1_name=b1_name+'_smooth',
                             b2_name=b2_name+'_smooth', **kwargs)
        else:
            print("Using previously computed flux function, may be for"
                  " different fields")

        db2_d1, db2_d2 = np.gradient(smooth_b2_data, *b2.mesh, axis=(1, 2))
        db1_d1, db1_d2 = np.gradient(smooth_b1_data, *b1.mesh, axis=(1, 2))
        fluxfn_hessian_det = db1_d2*(-db2_d1) - (-db2_d2)*db1_d1

        # Contour finding using skimage.measure.find_contours
        zeros_b2 = measure.find_contours(smooth_b2_data[time_idx], 0)
        zeros_b1 = measure.find_contours(smooth_b1_data[time_idx], 0)
        # Creating dictionary of interpolators over the indices of the mesh
        default_meshgrid = np.meshgrid(*b1.mesh, indexing='ij')
        all_pts = np.stack(default_meshgrid, axis=2)
        interps = {}
        idx_mesh = (np.array(range(len(b1.mesh[0]))),
                    np.array(range(len(b1.mesh[1]))))
        interps['all_pts'] = interp.RegularGridInterpolator(idx_mesh, all_pts)
        interps['fluxfn_hessian_det'] = interp.RegularGridInterpolator(
            idx_mesh, fluxfn_hessian_det[time_idx])
        interps['flux_fn'] = interp.RegularGridInterpolator(
            idx_mesh, self.variables['flux_fn'].data[time_idx])
        """ Finding intersections of zero contours"""
        # each contour is an m x 2 array for m some other number depending on
        # the number of points in the contour
        # break up each contour into m-1 line segments and check if
        # they intersect each other
        nulls_list = []
        for contour_2 in zeros_b2:
            endpt_2_1 = contour_2[:-1]
            endpt_2_2 = contour_2[1:]
            for contour_1 in zeros_b1:
                endpt_1_1 = contour_1[:-1]
                endpt_1_2 = contour_1[1:]
                # check if the 2 contour line segments intersect
                # any of the 1 contour ones
                for i in range(endpt_2_1.shape[0]):
                    endpt_2_1i = endpt_2_1[i].reshape(-1, 2)
                    endpt_2_2i = endpt_2_2[i].reshape(-1, 2)
                    # get indices of contour_2 which intercept
                    intersects = np.nonzero(intersect_true(endpt_1_1,
                                                           endpt_1_2,
                                                           endpt_2_1i,
                                                           endpt_2_2i))[0]
                    if len(intersects) != 0:  # only add in points that e2ist
                        intersect_pt = line_intersect(endpt_1_1[intersects],
                                                      endpt_1_2[intersects],
                                                      endpt_2_1i, endpt_2_2i)
                        # round intersections to nearest integer
                        nulls_list.append(intersect_pt)

        nulls = np.concatenate(nulls_list, axis=0)
        print("Number of nulls: ", len(nulls))

        # Not doing any sort of combining
        blobs_arr = nulls

        """ separate out the X and O points """
        o_idxs = [np.sign(interps['fluxfn_hessian_det'](blobs_arr[i])[0]) == 1
                  for i in range(blobs_arr.shape[0])]
        x_idxs = [np.sign(interps['fluxfn_hessian_det'](blobs_arr[i])[0]) == -1
                  for i in range(blobs_arr.shape[0])]
        o_coords = blobs_arr[o_idxs]
        x_coords = blobs_arr[x_idxs]
        self._add_param('x_coords', x_coords)
        self._add_param('o_coords', o_coords)

        """ define the separatrices and o-type structures """
        seps_mask = np.zeros_like(self.variables['flux_fn'].data[time_idx])
        o_structures = np.zeros_like(seps_mask)
        for i in range(x_coords.shape[0]):  # I love nested for loops...
            xline_contours = measure.find_contours(self.variables
                                                   ['flux_fn'].data[time_idx],
                                                   level=interps
                                                   ['flux_fn'](x_coords[i]))
            xpt_mask = np.zeros_like(seps_mask)
            xpt_mask[tuple(x_coords[i].astype(np.int64))] = 1
            xpt_mask = binary_dilation(xpt_mask).astype(np.int64)
            local_seps = np.zeros_like(seps_mask)
            local_seps = np.logical_or(local_seps, xpt_mask)
            for contour in xline_contours:
                if (min(np.linalg.norm(contour - x_coords[i], axis=1))
                        <= d_per_de*de_tol):
                    for point in contour:
                        local_seps[tuple(point.astype(np.int64))] = 1

            # fill in potential gaps in the contours
            local_seps_filled = gap_fill(local_seps)
            seps_mask = np.logical_or(seps_mask, local_seps_filled)

            # fill in all the sections according to their geometry
            while np.sum(local_seps_filled == 0) > 0:
                # select a random 0-valued point
                indices = np.nonzero(local_seps_filled == 0)
                random_int = rng.integers(len(indices[0]))
                random_idx = (indices[0][random_int], indices[1][random_int])
                structure_candidate = flood(local_seps_filled, random_idx,
                                            connectivity=1)
                local_seps_filled += -1*structure_candidate  # region
                if  (np.all(structure_candidate[0, :] == 0) and  # no collision
                     np.all(structure_candidate[-1, :] == 0) and     # w/ bdy
                     np.all(structure_candidate[:, 0] == 0) and
                     np.all(structure_candidate[:, -1] == 0)):
                    o_structures = np.logical_or(o_structures,
                                                 structure_candidate) \
                                                 .astype(np.int64)

        """ Find the current sheets """
        smooth_jy = nd.gaussian_filter(self.variables['jy'].data[time_idx],
                                       full_smoothing[1:])  # no time dim
        cs_maxes = nd.maximum_filter(smooth_jy, size=(10, 10))
        cs_mins = nd.minimum_filter(smooth_jy, size=(10, 10))
        max_thresh = cs_maxes.mean() + cs_maxes.std() * 3
        min_thresh = cs_mins.mean() - cs_mins.std() * 3

        # find areas greater/less than threshold value
        max_labels, max_num = nd.label(cs_maxes > max_thresh)
        min_labels, min_num = nd.label(cs_mins < min_thresh)
        # Get the positions of the extrema
        max_coords = nd.maximum_position(smooth_jy, labels=max_labels,
                                         index=np.arange(1, max_num + 1))
        min_coords = nd.minimum_position(smooth_jy, labels=min_labels,
                                         index=np.arange(1, min_num + 1))
        # Get the extrema values
        max_values = nd.maximum(smooth_jy, labels=max_labels,
                                index=np.arange(1, max_num + 1))
        min_values = nd.minimum(smooth_jy, labels=min_labels,
                                index=np.arange(1, min_num + 1))

        cs_loc_pos = np.zeros_like(smooth_jy)
        cs_loc_neg = np.zeros_like(smooth_jy)

        for i in range(max_num):
            # mask cs values which are "high enough"
            cs_pos = smooth_jy > max_values[i]*cs_limit
            # keep only cs which are local to the sheet
            cs_pos = flood(cs_pos, max_coords[i], connectivity=1)
            # update mask of all positive current sheets
            cs_loc_pos = np.logical_or(cs_loc_pos, cs_pos)

        for i in range(min_num):
            # mask cs values which are "low enough"
            cs_neg = smooth_jy < min_values[i]*.5
            # keep only cs which are local to the sheet
            cs_neg = flood(cs_neg, min_coords[i], connectivity=1)
            # update mask of all positive current sheets
            cs_loc_neg = np.logical_or(cs_loc_neg, cs_neg)

        """ Add in the variables """
        self._add_var('fluxfn_hessian_det', b1.timeseries, b1.mesh,
                      fluxfn_hessian_det)
        self._add_var('separatrices', b1.timeseries, b1.mesh,
                      seps_mask
                      .reshape((-1, *seps_mask.shape)))
        self._add_var('o_structures', b1.timeseries, b1.mesh,
                      o_structures.reshape((-1, *o_structures.shape)))
        self._add_var('current_sheets', b1.timeseries, b1.mesh,
                      (cs_loc_pos.astype(np.int64)
                       - cs_loc_neg.astype(np.int64))
                      .reshape((-1, *cs_loc_pos.shape)))
