
#################### Eurecon Preprocessing and Augmentation Script (ModelNet40 Train/Test Dataset Splits) ####################

# Python ENV:
# 1) Requires "python -m pip install tqdm'"
# 2) Requires latest version of PyTorch Geometric (https://github.com/rusty1s/pytorch_geometric)

# IMPORTANT: Original ModelNet40 dataset comes with a considerable amount of typos and errors in formatting within the files, 
# which is why it is important to fix those errors first. It is suggested to use the following script (https://github.com/cabraile/ModelNet40Fixer), 
# although you could use a custom fixer.
	
# HOW-TO:

# STEP 1: Define paths to dataset, Eurecon initialization script and tesselation axes

# STEP 2: Run the script within prepared environment with "python augment_ModelNet40.py > log.txt"

# STEP 3: Study the log.txt file for potential errors

# STEP 4: Use "chmod +x augment_ModelNet40.sh" to make the resulting generated filed executable

# STEP 5: Run the resulting executable file with "./augment_ModelNet40.sh" - script automatically places the augmented data samples in the same folder, where the original file can be found.

# As soon as you obtain an augmented dataset (258 531 .off files in case of 20 tesellation axes), you must use native MeshNet/Pytorch Geometric tools to further preprocess the dataset:

# 1) MeshNet (https://github.com/iMoonLab/MeshNet) - consider the "Data Preparation" pipeline subsection, offered by Feng et al. to reorganise and preprocess your augmented dataset.

# 2) PointNet++ (https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py) - consider using a native dataloader (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/modelnet.html#ModelNet)
# or make your own (https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html) to convert your augmented dataset with .off files to .pt file format, as stated in the manual.


import os
import glob
from tqdm import tqdm
from torch_geometric.io import read_off, write_off
from torch_geometric.transforms import NormalizeScale as NS

rmsd = 0.2
partition_parameter = 1
paths_train = paths_test = [] 
counter1 = counter2 = counter_after1 = counter_after2 = 0
scaling = NS()

python3_env_alias = 'python'
path_to_dataset = '/path/to/ModelNet40/folder'
path_to_eurecon = '/path/to/eurecon.py'
path_to_axes = '/path/to/tesselation_vertices_layer_0.txt'

file = open('augment_ModelNet40.sh', 'w')
path_list = os.listdir(path_to_dataset)

for path in path_list:
    paths_train.append(os.path.join(path_to_dataset, path, 'train'))
    paths_test.append(os.path.join(path_to_dataset, path, 'test'))

for object_path_train in paths_train:
    object_paths_train_list = [f for f in glob.glob(os.path.join(object_path_train, "*.off"), recursive=True)]
    for i in tqdm(object_paths_train_list, desc = 'Progress (Train Files)'):
        try:
            write_off(scaling(read_off(i)), i)
            out_string_train = str(str(python3_env_alias) + ' ' + str(path_to_eurecon) + ' -r ' + str(rmsd) + ' -p ' + str(partition_parameter) + ' -a ' + str(path_to_axes) + ' -i ' + str(i) + ' -o ' + str(object_path_train) + '\n')
            file.write(out_string_train)
        except:
            file.write('Failed to scale and write down the ' + str(i) + ' object...')
        
for object_path_test in paths_test:
    object_paths_test_list = [f for f in glob.glob(os.path.join(object_path_test, "*.off"), recursive=True)]
    for j in tqdm(object_paths_test_list, desc = 'Progress (Test Files)'):
        try:
            write_off(scaling(read_off(j)), j)
            out_string_test = str(str(python3_env_alias) + ' ' + str(path_to_eurecon) + ' -r ' + str(rmsd) + ' -p ' + str(partition_parameter) + ' -a ' + str(path_to_axes) + ' -i ' + str(j) + ' -o ' + str(object_path_test) + '\n')
            file.write(out_string_test)
        except:
            file.write('Failed to scale and write down the ' + str(j) + ' object...')
            
