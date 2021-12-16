# Eurecon

Equidistant and Uniform Data Augmentation for 3D Objects.

Python library for 3D data augmentation, based on Eurecon algorithm. 

- The library works with a variety of 3D file formats, including point cloud formats (.XYZ, .PTS, .PCD) and polygon mesh formats (.STL, .OFF, .OBJ, .PLY, .GLTF).
- Eurecon is computationally efficient, taking ~0.1 seconds to generate 1,000 samples  of an object of 1,000 3D points. 
- Based on numpy and open3d.
- Simple, flexible API that allows the library to be used in any machine learning pipeline.
- Large, diverse set of transformations based on the amount of input tessellation axes.
- Easy to extend the library to wrap around other libraries.
- Easy to extend to other tasks.



## Requirements:
- Python 3.6+
- click==7.0+
- numpy==1.17.2+
- open3d==0.10.0.0+
- tqdm==2.0.0+

Tests were performed on:

- Ubuntu 18.04
- Python 3.8
- click==7.1.2
- numpy==1.17.2
- open3d==0.10.0.0
- tqdm==4.46.1

## How to use:

Easiest way to run the augmentation via Eurecon is running eurecon.py with the following command line:

python eurecon.py -r <desired_RMSD> -p <desired_partition_parameter> -a <path/to/tessellation/axes/file> -i <path/to/input/files> -o <path/to/the/output/directory>

Detailed instructions on how to run Eurecon to reproduce an augmented ModelNet40 dataset are given in augment_ModelNet40.py script.

## Example:

python eurecon.py -r 0.5 -p 1 -a resources/tesselation_vertices_layer_0.txt -i examples/test.off -o examples/test_output
