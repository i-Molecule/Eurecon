"""Parser module."""
import copy
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from ..utils.metrics import timing
from .conformation import Conformation
from .transform import Transform

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 



class Parser:
    """Default class for Parser."""

    POINT_CLOUD = ("xyz", "pcd", "pts")
    TRIANGLE_MESH = ("stl", "obj", "off", "gltf", "ply")

    def __init__(self, output_directory):
        """Initialization."""
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory): os.mkdir(self.output_directory) 


    def parse_transform(self, axes_file_path, rmsd, partition):
        """
        The function which parses the given transform.

        Parameters:
            axes_file_path (String): Path to the file containing weights.
            rmsd (Float): RMSD value given as an input.
            partition (Float): Partition value given as an input.

        Returns:
            Transform (Transform class object): Transform object for the given input values.

        """

        axes = np.array(np.loadtxt(axes_file_path))

        return Transform(axes, rmsd, partition)

    @timing
    def parse_object(self, file_name):
        """
        The function to read and parse the given input file.

        Parameters:
            file_name (String): The name of the file to be parsed.

        Returns:
            data_object (varies on the input): Data object, specific for the inner libraries used
                for parsing the 3D object; used for writing the output later on.
            points_coords (A N-by-3 numpy array): An array of 3D object coordinates.
            object_length (Integer) : Number of points in a given 3D object.
            file_name (String): Name of the file.
        """
        if file_name.endswith(Parser.POINT_CLOUD):
            data_object = o3d.io.read_point_cloud(file_name)
            points_coords = np.transpose(np.asarray(data_object.points))
            object_length = len(points_coords[0])


        if file_name.endswith(Parser.TRIANGLE_MESH):
            data_object = o3d.io.read_triangle_mesh(file_name)
            points_coords = np.transpose(np.asarray(data_object.vertices))
            ### Scaling applied
            # points_coords = points_coords - points_coords.mean(axis=-2, keepdim=True)
            # scale = (1 / points_coords.abs().max()) * 0.999999
            # points_coords = (points_coords * scale).numpy()
            ###
            object_length = len(np.transpose(points_coords))
            return data_object, points_coords, object_length, file_name


    def parse_base_conformation(self, input_directory, weights_file_path=None):
        """
        The function which parses the given input file.

        Parameters:
            file_name (String): The name of the file to be parsed.
            weights_file_path (String, optional): Path to the file containing weights.

        Returns:
            coords (A N-by-3 numpy array): An array of 3D object coordinates.
            conformation (Conformation class object): Conformation object for the given input values.

        """
        if weights_file_path:
            weights = np.array(np.loadtxt(weights_file_path))
        else:
            weights = None
            data_obj, coords, object_length, file_name = self.parse_object(input_directory)
            conformation = Conformation(data_obj, coords, object_length, file_name, weights)
            self.conf_file_name = input_directory
            return coords, conformation

    @timing
    def write_conformation(
        self, data_object, new_conformation, data_file_name, counter
    ):
        """
        The function which generates a single conformation.

        Parameters:
            data_object (varies for the input): Data object, specific for the inner libraries used
                for parsing the 3D object; used for writing the output later on.
            new_conformation (A N-by-3 numpy array): A newly generated conformation
                for the given RMSD and tessellation axis.
            points_coords (A N-by-3 numpy array): An array of 3D object coordinates.
            data_file_name (String): Name of the file.
            counter (Integer): Iterable over the given tessellation axis array,
                placed as the name of the generated conformation.

        Returns:
            None.

        """
        format = data_file_name.split('.')[-1]
        new_file_name = f"{self.output_directory}/{data_file_name.split('/')[-1][:-(len(format)+1)]}_{counter}.{format}"

        
        if data_file_name.endswith(Parser.POINT_CLOUD):
            pcd_mod = o3d.geometry.PointCloud()
            pcd_mod.points = o3d.utility.Vector3dVector(new_conformation)
            o3d.io.write_point_cloud(new_file_name, pcd_mod)

        elif data_file_name.endswith(Parser.TRIANGLE_MESH):
            np_triangles = np.array(data_object.triangles)
            np_vertices = new_conformation
            mesh_mod_out = o3d.geometry.TriangleMesh()
            mesh_mod_out.vertices = o3d.utility.Vector3dVector(np_vertices)
            mesh_mod_out.triangles = o3d.utility.Vector3iVector(np_triangles)
            if format == "stl":
                mesh_mod_out.compute_vertex_normals()
            o3d.io.write_triangle_mesh(new_file_name, mesh_mod_out)

    def write_default(self, base_conformation: Conformation, coords):
        """
        The function writes out the base conformation to the given output directory.

        Parameters:
            base_conformation (Conformation class object):  Conformation object for the given input values.
            coords (A N-by-3 numpy array): An array of 3D object coordinates.

        Returns:
            None.

        """
        data_object = base_conformation.data_object
        data_file_name = base_conformation.data_file_name

        data_file_name = data_file_name.split("/")[-1]
        format = data_file_name.split(".")[-1]

        new_file_name = f"{self.output_directory}/{data_file_name}"

        if data_file_name.endswith(Parser.POINT_CLOUD):
            pcd_mod = o3d.geometry.PointCloud()
            pcd_mod.points = o3d.utility.Vector3dVector(coords.T)
            o3d.io.write_point_cloud(new_file_name, pcd_mod)

        elif data_file_name.endswith(Parser.TRIANGLE_MESH):
            np_triangles = np.array(data_object.triangles)
            np_vertices = np.array(data_object.vertices)
            mesh_mod_out = o3d.geometry.TriangleMesh()
            mesh_mod_out.vertices = o3d.utility.Vector3dVector(np_vertices)
            mesh_mod_out.triangles = o3d.utility.Vector3iVector(np_triangles)
            if format == "stl":
                mesh_mod_out.compute_vertex_normals()
            o3d.io.write_triangle_mesh(new_file_name, mesh_mod_out)

    def write_all_conformations(
        self, conformations: list, base_conformation: Conformation
    ):
        """
        The function writes out all of the generated conformations to the given output directory.

        Parameters:
            conformations (List):  List of all generated conformations.
            base_conformation (Conformation class object):  Conformation object for the given input values.

        Returns:
            None.

        """
        bar = tqdm(desc="Writing conformations", total=len(conformations))
        for counter, new_conformation in enumerate(conformations):
            self.write_conformation(
                base_conformation.data_object,
                new_conformation,
                base_conformation.data_file_name,
                counter
            )
            bar.update()
        bar.close()
