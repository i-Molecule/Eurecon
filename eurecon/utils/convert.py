import open3d as o3d
import numpy as np


# In[139]:

name = 80000
file_name = "/home/apm/Desktop/New test set/"+ str(name) +".stl" 




data_object = o3d.io.read_triangle_mesh(file_name)
points_coords = np.transpose(np.asarray(data_object.vertices))
object_length = len(np.transpose(points_coords))

pcd_mod = o3d.geometry.PointCloud()
pcd_mod.points = o3d.utility.Vector3dVector(points_coords.T)
o3d.io.write_point_cloud('/home/apm/Desktop/New test set/'+ str(object_length) +'.pcd', pcd_mod)
o3d.io.write_point_cloud('/home/apm/Desktop/New test set/'+ str(object_length) +'.xyz', pcd_mod)
o3d.io.write_point_cloud('/home/apm/Desktop/New test set/'+ str(object_length) +'.pts', pcd_mod)


np_triangles = np.array(data_object.triangles)
np_vertices = np.array(data_object.vertices)
mesh_mod_out = o3d.geometry.TriangleMesh()
mesh_mod_out.vertices = o3d.utility.Vector3dVector(np_vertices)
mesh_mod_out.triangles = o3d.utility.Vector3iVector(np_triangles)
o3d.io.write_triangle_mesh('/home/apm/Desktop/New test set/'+ str(object_length) +'.obj', mesh_mod_out)
o3d.io.write_triangle_mesh('/home/apm/Desktop/New test set/'+ str(object_length) +'.off', mesh_mod_out)
o3d.io.write_triangle_mesh('/home/apm/Desktop/New test set/'+ str(object_length) +'.gltf', mesh_mod_out)
o3d.io.write_triangle_mesh('/home/apm/Desktop/New test set/'+ str(object_length) +'.ply', mesh_mod_out)


print(object_length)


# In[ ]:




