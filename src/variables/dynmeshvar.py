from dataframework.src.variables.variable import Variable
# import numpy as np
# import scipy.interpolate

# Doesn't do anything right now
# but exists to be a nice empty flowerpot


class DynMeshVar(Variable):
    """ Class for representations of specific measured scalar values
        from simulations, whose measuring point is NOT fixed.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, label, timeseries, mesh, data):
        """ Initializes the DynMeshVar object """

        super().__init__(label, timeseries, mesh, data)

        raise ValueError(f'{self.__class__} is not implemented yet, sorry.')

    def _timeslice(self):
        pass

    def _zoom(self):
        pass

    def _spaceslice(self):
        pass
