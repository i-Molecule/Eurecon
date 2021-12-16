"""Base class for import."""
import datetime
import os

from .base.conformation import Conformation
from .base.ensemble import Ensemble
from .base.transform import Transform
from .base.parser import Parser

OPEN_3D_FORMATS = ("obj", "off", "ply", "gltf", "stl", "pcd", "pts", "xyz")


class Eurecon:
    """Base module class."""

    def __init__(
        self,
        input_directory: str,
        axes_file_path: str,
        output_directory: str,
        rmsd: float,
        partition: float,
        stdout_mode: bool = False,
        weights_file_path: bool = None,
        debug_mode: bool = False
    ):
        """Eurecon initialization."""
        self.input_directory: str = input_directory
        self.axes_file_path: str = axes_file_path
        self.weights_file_path: str = weights_file_path
        self.rmsd: float = rmsd
        self.output_directory: str = output_directory
        self.partition: float = partition
        self.stdout_mode: bool = stdout_mode
        self.debug_mode: bool = debug_mode

    def validate(self):
        """Parametres validation."""
        if not os.path.exists(self.input_directory):
            raise FileExistsError("Input file was not found.")
        if not os.path.exists(self.axes_file_path):
            raise FileExistsError("File with axes was not found.")
        if not os.path.exists(self.output_directory):
            raise Exception("Output directory was not found.")
        if not (0 <= self.partition <= 1):
            raise Exception("Given partition parameter did not lie within a range of [0, 1].")
        if self.weights_file_path is not None and not os.path.exists(self.weights_file_path):
            raise FileExistsError("File with weights was not found.")
        if self.rmsd <= 0:
            raise Exception("RMSD cannot be negative.")
    
    def start(self):
        """Start eurecon algorithm."""
        #self.validate()
        start = datetime.datetime.now()
        parser: Parser = Parser(self.output_directory)
        coords, base_conformation = parser.parse_base_conformation(
            self.input_directory, self.weights_file_path
        )
 

        transform: Transform = parser.parse_transform(
            self.axes_file_path, self.rmsd, self.partition
        )

        ensemble = Ensemble(base_conformation, transform, parser, self.debug_mode)


        start_no_parsing = datetime.datetime.now()
        ensemble.generate_ensemble(self.stdout_mode)

############################# Comment/Uncomment to disable/enable writedown ##############################
        if not self.stdout_mode:
            parser.write_default(base_conformation, coords)
            ensemble.write()
############################# Comment/Uncomment to disable/enable writedown ##############################

        total = datetime.datetime.now() - start
        total_no_parsing = datetime.datetime.now() - start_no_parsing

        return (
            parser.metrics,
            base_conformation.metrics,
            ensemble.metrics,
            {
                "total": total,
                "total_no_parsing": total_no_parsing.total_seconds(),
                "object_length": base_conformation.object_length,
                "axes_amount": len(transform.axes),
            },
        )
