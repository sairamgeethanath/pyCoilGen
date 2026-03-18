import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

radius = 0.07
nr = 30 # number of radial divisions
nt = 90 # 180 for 2 degree resolution, 90 for 4 degree resolution
inner_radius = radius / nr
z_positions = [-0.03675, 0.03675]
filename = "data/pyCoilGenData/Geometry_Data/Tenacity_circular_1.stl"


def create_mesh(radius,nr,nt,z):

    verts=[]
    faces=[]

    for i in range(nr):

        r = inner_radius + (radius-inner_radius)*i/(nr-1)

        for j in range(nt):

            theta=2*np.pi*j/nt

            x=r*np.cos(theta)
            y=r*np.sin(theta)

            verts.append([x,y,z])

    verts=np.array(verts)

    def vid(i,j):
        return i*nt+j

    for i in range(nr-1):

        for j in range(nt):

            v1=vid(i,j)
            v2=vid(i,(j+1)%nt)
            v3=vid(i+1,(j+1)%nt)
            v4=vid(i+1,j)

            faces.append([v1,v2,v3])
            faces.append([v1,v3,v4])

    return verts,np.array(faces)


def write_stl(filename,verts,faces):

    for i,tri in enumerate(faces):

        v1,v2,v3 = verts[tri]

        normal = np.cross(v2-v1, v3-v1)

        if normal[2] < 0:  # flip triangle
            faces[i] = [tri[0], tri[2], tri[1]]

    with open(filename,"wb") as f:

        f.write(b'PythonSTL'+b' '*(80-len('PythonSTL')))
        f.write(struct.pack("<I",len(faces)))

        for tri in faces:

            v1,v2,v3=verts[tri]

            normal=np.cross(v2-v1,v3-v1)
            norm=np.linalg.norm(normal)

            if norm==0:
                n=[0,0,0]
            else:
                n=normal/norm

            f.write(struct.pack("<3f",*n))

            for v in [v1,v2,v3]:
                f.write(struct.pack("<3f",*v))

            f.write(struct.pack("<H",0))


# ==========================================================
# CHECK MESH QUALITY
# ==========================================================
def check_mesh_quality(mesh, dsv_radius = 0.016, visualize = False):
    print("Mesh quality check:")
    print("Bounds:", mesh.bounds)
    print("Centroid:", mesh.centroid)
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], s=1)
        ax.set_title("STL Mesh Visualization")

        # Plot the DSV sphere for reference
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = dsv_radius * np.cos(u)
        y = dsv_radius * np.sin(u) * np.cos(v)
        z = dsv_radius * np.sin(u) * np.sin(v)
        ax.plot_wireframe(x, y, z, color='r', alpha=0.5, label='DSV Sphere')
        ax.legend()     
        plt.show()
        


def create_stl_mesh(radius,nr,nt,z_positions,filename):
    all_verts=[]
    all_faces=[]
    offset=0
    for z in z_positions:

        v,f=create_mesh(radius,nr,nt,z)

        all_verts.append(v)
        all_faces.append(f+offset)

        offset+=len(v)

    verts=np.vstack(all_verts)
    faces=np.vstack(all_faces)

    write_stl(filename,verts,faces)

    print("STL saved:",filename)