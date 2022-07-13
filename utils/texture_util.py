import os
import numpy as np


def write_obj_pair(file_name1, file_name2, verts1, faces1, verts2, faces2, p2p, texture_file):
    # write off for shape 1
    uv1 = generate_tex_coords(verts1)
    write_obj_with_texture(verts1, faces1, file_name1, uv1, texture_file)

    # write off for shape 2
    uv2 = uv1[p2p]
    write_obj_with_texture(verts2, faces2, file_name2, uv2, texture_file)


def generate_tex_coords(verts, col1=1, col2=0, mult_const=1):
    ind = np.argsort(np.std(verts, axis=0))[::-1]
    verts = verts[:, ind]
    vt = np.stack([verts[:, col1], verts[:, col2]], axis=-1)
    vt -= np.min(vt, axis=0, keepdims=True)
    vt = mult_const * vt / np.max(vt)
    vt[:, 0] = (vt[:, 0] - vt[:, 0].min()) / (vt[:, 0].max() - vt[:, 0].min())
    vt[:, 1] = (vt[:, 1] - vt[:, 1].min()) / (vt[:, 1].max() - vt[:, 1].min())
    return vt


def write_obj_with_texture(verts, faces, file_name, uv=None, texture_name='texture.jpg'):
    """
    write .obj file with texture.
    Args:
        verts (np.ndarray): vertices. [V, 3].
        faces (np.ndarray): faces. [F, 3].
        uv (np.ndarray, None): texture maps. [V, 2]
        texture_name (str): texture map image file name.
        file_name (str): stored file name.
    """
    assert verts.shape[-1] == 3, f'vertex does not have the correct format: {verts.shape}.'
    assert faces.shape[-1] == 3, f'face does not have the correct format: {faces.shape}.'
    if uv is not None:
        assert uv.shape[-1] == 2, f'vertex texture does not have the correct format: {uv.shape}.'
    object_name = os.path.splitext(os.path.basename(file_name))[0]
    faces = faces.astype(int)
    with open(file_name, 'w') as f:
        # head
        f.write('# write_obj (c) 2004 Gabriel Peyr\n')
        f.write(f'mtllib ./{object_name}.mtl\n')
        f.write(f'g\n# object {object_name} to come\n')

        # vertex position
        f.write(f'# {verts.shape[0]} vertex\n')
        for i in range(verts.shape[0]):
            f.write(f'v {verts[i][0]:.6f} {verts[i][1]:.6f} {verts[i][2]:.6f}\n')

        # use mtl
        f.write(f'g {object_name}_export\n')
        mtl_bump_name = 'material_0'
        f.write(f'usemtl {mtl_bump_name}\n')

        # face
        f.write(f'# {faces.shape[0]} faces\n')
        faces += 1
        for i in range(faces.shape[0]):
            f.write(f'f {faces[i][0]}/{faces[i][0]} {faces[i][1]}/{faces[i][1]} {faces[i][2]}/{faces[i][2]}\n')

        # vertex texture
        if uv is not None:
            for i in range(uv.shape[0]):
                f.write(f'vt {uv[i][0]:.6f} {uv[i][1]:.6f}\n')
        else:
            vertext = verts[:, 0:2] * 0 - 1
            # vertex position
            f.write(f'# {vertext.shape[0]} vertex texture\n')
            for i in range(vertext.shape[0]):
                f.write(f'vt {vertext[i][0]:.6f} {vertext[i][1]:.6f}\n')

    # generate MTL file
    if uv is not None:
        mtl_file = file_name.replace('.obj', '.mtl')
        Ka = [0.2, 0.2, 0.2]
        Kd = [1, 1, 1]
        Ks = [1, 1, 1]
        Tr = 1
        Ns = 0
        illum = 2
        with open(mtl_file, 'a') as f:
            f.write('# write_obj (c) 2004 Gabriel Peyr\n')
            f.write(f'newmtl {mtl_bump_name}\n')
            f.write(f'Ka  {Ka[0]:.6f} {Ka[1]:.6f} {Ka[2]:.6f}\n')
            f.write(f'Kd  {Kd[0]:.6f} {Kd[1]:.6f} {Kd[2]:.6f}\n')
            f.write(f'Ks  {Ks[0]:.6f} {Ks[1]:.6f} {Ks[2]:.6f}\n')
            f.write(f'Tr  {Tr}\n')
            f.write(f'Ns  {Ns}\n')
            f.write(f'illum {illum}\n')
            f.write(f'map_Kd {texture_name}\n')
            f.write('#\n# EOF\n')
