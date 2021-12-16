"""Transform module."""
import numpy as np


class Transform:
    """Default class for Transform."""

    def __init__(
        self, axes, rmsd, partition, normalized=False,
    ):
        """Transform initialization."""
        self.rmsd = rmsd
        self.partition = partition
        if not normalized:
            self.axes = self.normalize_axes(axes)
        else:
            self.axes = axes

    def normalize_axes(self, a):
        """
        The function to perform normalization for a given set of axes.

        Parameters:
            a (A N-by-3 numpy array): An axis array to be normalized.

        Returns:
            axes_array (A N-by-3 numpy array): A normalized axis array.
        """
        axes_array = a / np.linalg.norm(a, axis=1, keepdims=True)
        return axes_array
