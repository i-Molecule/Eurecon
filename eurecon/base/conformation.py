"""Conformation module."""
import numpy as np

from ..utils.metrics import timing


class Conformation:
    """Default class for Conformation."""

    def __init__(self, data_object, coords, object_length, data_file_name, weights):
        """Conformation initialization."""
        self.data_object = data_object
        self.object_length: int = object_length
        if self.object_length == None:
            raise ValueError("Object length cannot be equal to zero.")
        self.data_file_name: str = data_file_name
        if self.data_file_name == None:
            raise Exception("File name cannot be empty")
        self.weights_are_ones: bool = False

        if not weights:
            weights = np.ones(object_length)
            self.weights_are_ones = True
        self.weights = weights
        if np.isnan(self.weights).any():
            raise ValueError('Weights array contains NaN entries.')
        self.total_weight = np.sum(weights)

        self.coords_in_center_of_mass = self.get_coords_moved_to_mass_center(coords)
        self.inertia_tensor = self.calc_inertia_tensor()

    @timing
    def calc_inertia_tensor(self):
        """
        The function to calculate the inertia tensor for a given set of points and their corresponding masses.
        Parameters:
            None

        Returns:
            inertia_matrix (An 3-by-3 numpy array): A 3-by-3 inertia tensor.
        """
        a = self.coords_in_center_of_mass.T
        if self.weights_are_ones:
            inertia_matrix = np.array(
                [
                    [
                        np.sum(np.sum([np.square(a[1]), np.square(a[2]),], axis=0)),
                        -np.sum(a[0] * a[1], axis=0),
                        -np.sum(a[0] * a[2], axis=0),
                    ],
                    [
                        -np.sum(a[0] * a[1], axis=0),
                        np.sum(np.sum([np.square(a[0]), np.square(a[2]),], axis=0)),
                        -np.sum(a[1] * a[2], axis=0),
                    ],
                    [
                        -np.sum(a[0] * a[2], axis=0),
                        -np.sum(a[1] * a[2], axis=0),
                        np.sum(np.sum([np.square(a[0]), np.square(a[1]),], axis=0)),
                    ],
                ]
            )
        else:
            inertia_matrix = np.array(
                [
                    [
                        np.sum(
                            self.weights
                            * np.sum([np.square(a[1]), np.square(a[2]),], axis=0)
                        ),
                        -np.sum(self.weights * a[0] * a[1], axis=0),
                        -np.sum(self.weights * a[0] * a[2], axis=0),
                    ],
                    [
                        -np.sum(self.weights * a[0] * a[1], axis=0),
                        np.sum(
                            self.weights
                            * np.sum([np.square(a[0]), np.square(a[2]),], axis=0)
                        ),
                        -np.sum(self.weights * a[1] * a[2], axis=0),
                    ],
                    [
                        -np.sum(self.weights * a[0] * a[2], axis=0),
                        -np.sum(self.weights * a[1] * a[2], axis=0),
                        np.sum(
                            self.weights
                            * np.sum([np.square(a[0]), np.square(a[1]),], axis=0)
                        ),
                    ],
                ]
            )
        return inertia_matrix

    def calc_mass_center(self, coords):
        """
        The function to calculate the center of masses for a given set of points and their corresponding masses.

        Parameters:
          coords (A N-by-3 numpy array): An array of 3D object coordinates.

        Returns:
          A 3-by-1 vector, containing center of mass coordinates.
        """
        if self.weights_are_ones:
            return (
            np.array([np.sum(coords, axis=0)]).T
                / self.total_weight
            )
        else:
            return (
                np.array(
                    [
                        np.sum(coords, axis=0) * self.weights
                    ]
                ).T
                / self.total_weight
            )

    @timing
    def get_coords_moved_to_mass_center(self, coords):
        """
        The function to relocate the center of mass of a given 3D object frame to the origin.

        Parameters:
          coords (A N-by-3 numpy array): An array of 3D object coordinates.

        Returns:
          coords_in_center_of_mass_transposed (A N-by-1 numpy array): An array, containing the coordinates of the relocated 3D object frame.
        """
        self.center_of_mass = self.calc_mass_center(coords)

        coords_in_center_of_mass_transposed = coords.T - self.center_of_mass
        return coords_in_center_of_mass_transposed
