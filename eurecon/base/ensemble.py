"""Ensemble module."""
import numpy as np
from tqdm import tqdm
from .parser import Parser
from ..utils.metrics import timing
from .conformation import Conformation
from .transform import Transform


class Ensemble:
    """Default class for Ensemble."""

    def __init__(
        self,
        base_conformation: Conformation,
        transform: Transform,
        parser: Parser,
        debug_mode: bool
    ):
        """Ensemble initialization."""
        self.base_conformation = base_conformation
        self.transform = transform

        self.parser = parser
        self.object_length = len(self.base_conformation.coords_in_center_of_mass[0])

        self.conformations = []
        self.debug_mode = debug_mode

    def count_rotation_matrix(self, axis, theta):
        """
        The function to calculate a rotation matrix using quaternion multiplication.

        Parameters:
            axis (A 3-by-1 numpy array):
                The axis, around which the rotation is to be performed.
            theta (Scalar):
                Angle (in degrees), by which the desired 3D object
                    frame would be rotated around given axis.

        Returns:
            matrix (A 3-by-3 numpy array): Rotation matrix.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        matrix = np.array(
            [
                [
                    axis[0] * axis[0] * (1.0 - c) + c,
                    axis[0] * axis[1] * (1.0 - c) - axis[2] * s,
                    axis[0] * axis[2] * (1.0 - c) + axis[1] * s,
                ],
                [
                    axis[1] * axis[0] * (1.0 - c) + axis[2] * s,
                    axis[1] * axis[1] * (1.0 - c) + c,
                    axis[1] * axis[2] * (1.0 - c) - axis[0] * s,
                ],
                [
                    axis[2] * axis[0] * (1.0 - c) - axis[1] * s,
                    axis[2] * axis[1] * (1.0 - c) + axis[0] * s,
                    axis[2] * axis[2] * (1.0 - c) + c,
                ],
            ]
        )

        return matrix

    @timing
    def rmsd_angle(self, chosen_axis):
        """
        The function to calculate the rotation angle for a given combination of partition and RMSD parameters.

        Parameters:
            chosen_axis (A 3-by-1 numpy array): The axis, which serves as a basis for performing rotation calculations.

        Returns:
            trans_angle (Float): The rotation angle, corresponding to the given partition and RMSD.
            partition (Float): Redefined partition parameter,
                changes in case there is no suitable rotation angle for a given set of parameters.
        """
        partition = self.transform.partition

        ntn = np.dot(
            chosen_axis, np.dot(self.base_conformation.inertia_tensor, chosen_axis.T)
        )
        in_arcsin = np.sqrt(
            np.power(self.transform.rmsd, 2)
            * partition
            * self.base_conformation.total_weight
            / (4 * ntn)
        )
        if in_arcsin > 1:
            trans_angle = np.pi
            partition = (4 / self.base_conformation.total_weight * ntn) / np.power(
                self.transform.rmsd, 2
            )
            print(
                "Warning: arcsin exceeds 1 for the given conformation, replacing with pi instead."
            )
        else:
            trans_angle = 2 * np.arcsin(in_arcsin)

        return trans_angle, partition

    @timing
    def generate_conformation(self, chosen_axis):
        """
        The function to generate a conformation, which was transformed using the given partition and RMSD parameters.

        Parameters:
            chosen_axis (A 3-by-1 numpy array):
                The axis, which serves as a basis for performing rotation calculations.

        Returns:
            conformation (An N-by-3 numpy array):
                3D object point array, which was transformed using the given partition and RMSD parameters.
        """
        angle, partition = self.rmsd_angle(chosen_axis)
        rotation_matrix = self.count_rotation_matrix(chosen_axis, angle)

        conformation = np.sum(
            [
                np.matmul(
                    self.base_conformation.coords_in_center_of_mass, rotation_matrix
                ),
                self.transform.rmsd * np.sqrt(1 - partition) * chosen_axis,
            ],
            axis=0,
        )
        if self.debug_mode:
            self.check_rmsd(conformation)
        conformation = conformation + self.base_conformation.center_of_mass

        return conformation

    
    @timing
    def check_rmsd(self, conformation):
        """
        The function which performs trivial RMSD calculation for comparison with Eurecon.

        Parameters:
            conformation (An N-by-3 numpy array): 3D object point array,
                which was transformed using the given partition and RMSD parameters.

        Returns:
            None.
        """
        trivial_rmsd = self.calc_trivial_rmsd(conformation)

        if (
            not self.transform.rmsd - 0.00001
            < trivial_rmsd
            < self.transform.rmsd + 0.00001
        ):
            raise ValueError('Wrong RMSD')

    def calc_trivial_rmsd(self, a):
        """
        The function which calculates trivial RMSD.

        Parameters:
            a (An N-by-3 numpy array): 3D object point array,
                which was transformed using the given partition and RMSD parameters.

        Returns:
            RMSD value.
        """
        b = self.base_conformation.coords_in_center_of_mass
        return np.sqrt(
            (1 / self.base_conformation.object_length) * np.sum((a - b) ** 2)
        )

    def generate_ensemble(self, stdout_mode: bool = True):
        """
        The function which generates an ensemble of conformations.

        Parameters:
            stdout_mode (Boolean): Parameter which allows/prohibits writing conformation ensembles into files.

        Returns:
            None.
        """

        POINT_CLOUD = ("pcd", "pts", "xyz")
        TRIANGLE_MESH = ("stl", "obj", "off", "gltf", "ply")
        object_type = str(self.base_conformation.data_file_name)[:-3]
        if object_type in POINT_CLOUD:
            print_type = "Point Cloud"
        else:
            print_type = "Mesh"
        
            print('Augmenting ' + str(self.base_conformation.data_file_name) + ' object' + '  |  Type: ' + str(print_type) + '  | RMSD Value: ' + str(self.transform.rmsd))
            

        name_of_bar = 'Processing conformations' if stdout_mode else 'Generating conformations'
        bar = tqdm(total=len(self.transform.axes), desc=name_of_bar)

        for counter, axis in enumerate(self.transform.axes):
            new_conformation = self.generate_conformation(axis)
            if stdout_mode:
                self.parser.write_conformation(
                    self.base_conformation.data_object,
                    new_conformation,
                    self.base_conformation.data_file_name,
                    counter,
                    self.base_conformation.object_length,
                )
            else:
                self.conformations.append(new_conformation)
            bar.update()

        bar.close()

    def write(self):
        """
        The function which performs the file generation.

        Parameters:
            None.

        Returns:
            None.
        """

        self.parser.write_all_conformations(self.conformations, self.base_conformation)
