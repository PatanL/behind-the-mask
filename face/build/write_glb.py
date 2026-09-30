"""Stage 3: assemble web/public/face/android.glb from the cached mesh + baked textures.

Run: face/build/.venv/bin/python face/build/write_glb.py
"""
import os, json, io
import numpy as np
from gltf import GLB, ELEMENT_ARRAY_BUFFER
from ict import CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'web', 'public', 'face', 'android.glb')
TEX = os.path.join(HERE, '..', 'art_raw', 'tex')

# skeleton pivots (metres, model space) -- head pivot ~ occipital condyles, neck pivot ~ C7
HEAD_PIVOT = np.array([0.0, -0.015, -0.010])
NECK_PIVOT = np.array([0.0, -0.120, -0.015])

POS_EPS = 2e-6   # 0.002 mm -- sparse threshold for morph position deltas
NRM_EPS = 2e-3


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def head_weight(p):
    # slanted head/neck junction: s = y + 0.45 z (in cm); blend across the neck
    y, z = p[:, 1] * 100, p[:, 2] * 100
    return smoothstep(-9.5, -5.0, y + 0.45 * z)


def quat_from_to(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    w = 1 + a @ b
    q = np.array([v[0], v[1], v[2], w])
    return q / np.linalg.norm(q)


def quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v * (w * w - u @ u) + 2 * u * (u @ v) + 2 * w * np.cross(u, v)


def main():
    d = np.load(os.path.join(CACHE, 'mesh.npz'), allow_pickle=True)['d'].item()
    names = d['morph_names']
    g = GLB()
    g.ext_used.add('KHR_mesh_quantization')
    g.ext_req.add('KHR_mesh_quantization')

    tex = {}
    for key in ['skin_seams', 'skin_ao']:
        p = os.path.join(TEX, key + '.png')
        if os.path.exists(p):
            tex[key] = g.image_png(open(p, 'rb').read(), key)

    # ---------------------------------------------------------- materials
    def porcelain(name, extra=None):
        m = {'name': name,
             'pbrMetallicRoughness': {'baseColorFactor': [0.93, 0.905, 0.87, 1.0], 'metallicFactor': 0.0, 'roughnessFactor': 0.42},
             'extensions': {'KHR_materials_clearcoat': {'clearcoatFactor': 1.0, 'clearcoatRoughnessFactor': 0.07}},
             'extras': {'role': 'porcelain'}}
        if extra:
            m['extras'].update(extra)
        return g.material(m)

    seam_extra = {}
    if 'skin_seams' in tex:
        seam_extra['seamTexture'] = tex['skin_seams']
    mat_face = porcelain('Porcelain', seam_extra)
    mat_socket = g.material({'name': 'Cavity', 'pbrMetallicRoughness': {'baseColorFactor': [0.035, 0.04, 0.05, 1], 'metallicFactor': 0, 'roughnessFactor': 0.55},
                             'extras': {'role': 'cavity'}})
    mat_teeth = g.material({'name': 'Teeth', 'pbrMetallicRoughness': {'baseColorFactor': [0.86, 0.86, 0.84, 1], 'metallicFactor': 0, 'roughnessFactor': 0.3},
                            'extensions': {'KHR_materials_clearcoat': {'clearcoatFactor': 0.6, 'clearcoatRoughnessFactor': 0.1}},
                            'extras': {'role': 'teeth'}})
    mat_gums = g.material({'name': 'Gums', 'pbrMetallicRoughness': {'baseColorFactor': [0.10, 0.11, 0.13, 1], 'metallicFactor': 0, 'roughnessFactor': 0.45},
                           'extras': {'role': 'gums'}})
    em = d['eye_mesh']
    mat_eye = g.material({'name': 'Eye', 'pbrMetallicRoughness': {'baseColorFactor': [0.9, 0.92, 0.95, 1], 'metallicFactor': 0, 'roughnessFactor': 0.1},
                          'extensions': {'KHR_materials_clearcoat': {'clearcoatFactor': 1.0, 'clearcoatRoughnessFactor': 0.02}},
                          'extras': {'role': 'eye', 'R': em['R'], 'apex': em['apex'], 'corneaZ0': em['cornea_z0'], 'corneaR': em['cornea_r'],
                                     'limbusZ': em['limbus_z'], 'limbusR': em['limbus_r']}})

    stats = {'tris': 0, 'verts': 0, 'sparse': 0}

    # ---------------------------------------------------------- morph meshes
    def build_mesh(mesh_name, prim_keys, mats, skinned=False, offset=np.zeros(3)):
        used = [n for n in names if any(n in d['prims'][k]['morph_pos'] for k in prim_keys)]
        prims = []
        for k, mat in zip(prim_keys, mats):
            p = d['prims'][k]
            pos = (p['pos'] - offset).astype(np.float32)
            attrs = {'POSITION': g.accessor(pos, minmax=True),
                     'NORMAL': g.accessor(p['nrm'].astype(np.float32)),
                     'TEXCOORD_0': g.accessor(uv_atlas(k, p['uv']))}
            if skinned:
                w = head_weight(p['pos'])
                joints = np.zeros((len(pos), 4), np.uint8)
                joints[:, 1] = 1
                weights = np.zeros((len(pos), 4), np.uint8)
                wh = np.round(w * 255).astype(np.int32)
                weights[:, 1] = wh
                weights[:, 0] = 255 - wh
                attrs['JOINTS_0'] = g.accessor(joints)
                attrs['WEIGHTS_0'] = g.accessor(weights, normalized=True)
            targets = []
            for n in used:
                if n in p['morph_pos']:
                    pa, cnt = g.sparse_accessor(p['morph_pos'][n], POS_EPS, quant=True, minmax=True)
                    na, cnt2 = g.sparse_accessor(p['morph_nrm'][n], NRM_EPS, quant=True)
                    stats['sparse'] += cnt + cnt2
                else:
                    pa, _ = g.sparse_accessor(np.zeros((len(pos), 3)), minmax=True)
                    na, _ = g.sparse_accessor(np.zeros((len(pos), 3)))
                targets.append({'POSITION': pa, 'NORMAL': na})
            prim = {'attributes': attrs, 'indices': g.accessor(p['idx'], target=ELEMENT_ARRAY_BUFFER), 'material': mat}
            if targets:
                prim['targets'] = targets
            prims.append(prim)
            stats['tris'] += len(p['idx'])
            stats['verts'] += len(pos)
        m = {'name': mesh_name, 'primitives': prims}
        if used:
            m['weights'] = [0.0] * len(used)
            m['extras'] = {'targetNames': used}
        return g.mesh(m)

    # ---------------------------------------------------------- nodes
    root = g.node({'name': 'AndroidFace', 'children': []}, root=True)
    neck = g.node({'name': 'neck', 'translation': NECK_PIVOT.tolist(), 'children': []})
    head = g.node({'name': 'head', 'translation': (HEAD_PIVOT - NECK_PIVOT).tolist(), 'children': []})
    g.j['nodes'][root]['children'].append(neck)
    g.j['nodes'][neck]['children'].append(head)

    skin_mesh = build_mesh('Skin', ['face', 'back', 'socket'], [mat_face, mat_face, mat_socket], skinned=True)
    ibm = np.zeros((2, 16), np.float32)
    for i, pv in enumerate([NECK_PIVOT, HEAD_PIVOT]):
        m = np.eye(4)
        m[:3, 3] = -pv
        ibm[i] = m.T.ravel()  # column-major
    ibm_acc = g.accessor(ibm, target=None)
    g.j['accessors'][ibm_acc]['type'] = 'MAT4'
    g.j['accessors'][ibm_acc]['count'] = 2
    g.j['skins'] = [{'name': 'NeckHead', 'joints': [neck, head], 'skeleton': neck, 'inverseBindMatrices': ibm_acc}]
    skin_node = g.node({'name': 'Skin', 'mesh': skin_mesh, 'skin': 0})
    g.j['nodes'][root]['children'].append(skin_node)

    teeth_mesh = build_mesh('Teeth', ['teeth'], [mat_teeth], offset=HEAD_PIVOT)
    gums_mesh = build_mesh('Mouth', ['gums'], [mat_gums], offset=HEAD_PIVOT)
    for nm, mi in [('Teeth', teeth_mesh), ('Mouth', gums_mesh)]:
        g.j['nodes'][head]['children'].append(g.node({'name': nm, 'mesh': mi}))

    # eyes: node at eyeball centre, local +Z = rest optical axis
    eye_mesh = g.mesh({'name': 'Eyeball', 'primitives': [{
        'attributes': {'POSITION': g.accessor(em['pos'], minmax=True), 'NORMAL': g.accessor(em['nrm'])},
        'indices': g.accessor(em['idx'], target=ELEMENT_ARRAY_BUFFER), 'material': mat_eye}]})
    stats['tris'] += 2 * len(em['idx'])
    stats['verts'] += 2 * len(em['pos'])
    for s in 'LR':
        e = d['eyes'][s]
        q = quat_from_to(np.array([0, 0, 1.0]), np.array(e['axis']))
        g.j['nodes'][head]['children'].append(g.node({
            'name': 'Eye' + s, 'mesh': eye_mesh,
            'translation': (np.array(e['center']) - HEAD_PIVOT).tolist(),
            'rotation': q.tolist(),
            'extras': {'restAxis': e['axis']}}))

    g.j['asset']['extras'] = {'source': 'ICT-FaceKit (MIT) generic neutral + expression shapes; see face/README.md',
                              'morphTargets': names}
    size = g.write(OUT)
    print(f'wrote {OUT}: {size/1e6:.2f} MB, tris={stats["tris"]}, verts={stats["verts"]}, sparse rows={stats["sparse"]}, morphs={len(names)}')


def uv_atlas(prim_key, uv):
    """ICT uses UDIM tiles (face u in [0,1], back head in [1,2]). Pack both into one 2:1 atlas;
    glTF v points down, so flip."""
    uv = uv.astype(np.float64).copy()
    if prim_key in ('face', 'back', 'socket'):
        uv[:, 0] = uv[:, 0] / 2.0
    else:
        uv[:, 0] = uv[:, 0] - np.floor(uv[:, 0].min())
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv.astype(np.float32)


if __name__ == '__main__':
    main()
