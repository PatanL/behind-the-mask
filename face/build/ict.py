"""Shared helpers for reading the ICT-FaceKit (MIT) FaceXModel OBJ files."""
import os, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ICT_DIR = os.environ.get('ICT_DIR', os.path.join(HERE, '..', 'art_raw', 'ict', 'FaceXModel'))
CACHE = os.path.join(HERE, '..', 'art_raw', 'cache')

# Geometry ranges from the ICT-FaceKit README (vertex index ranges, inclusive)
REGIONS = {
    'face':        (0, 9408),
    'headneck':    (9409, 11247),
    'mouthsocket': (11248, 13293),
    'eyesocketL':  (13294, 13677),
    'eyesocketR':  (13678, 14061),
    'gums':        (14062, 17038),
    'teeth':       (17039, 21450),
    'eyeballL':    (21451, 23020),
    'eyeballR':    (23021, 24590),
    'lacrimalL':   (24591, 24794),
    'lacrimalR':   (24795, 24998),
    'eyeblendL':   (24999, 25022),
    'eyeblendR':   (25023, 25046),
    'occlusionL':  (25047, 25198),
    'occlusionR':  (25199, 25350),
    'lashesL':     (25351, 26034),
    'lashesR':     (26035, 26718),
}
IRIS = {'L': (22221, 23020), 'R': (23791, 24590)}
SCLERA = {'L': (21451, 22220), 'R': (23021, 23790)}


def region_of(vidx):
    for k, (a, b) in REGIONS.items():
        if a <= vidx <= b:
            return k
    return None


def read_obj(path, want_faces=False):
    vs, vts, faces = [], [], []
    mats = []
    cur = None
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                vs.append(line[2:].split()[:3])
            elif want_faces:
                if line.startswith('vt '):
                    vts.append(line[3:].split()[:2])
                elif line.startswith('usemtl '):
                    cur = line.split()[1]
                elif line.startswith('f '):
                    idx = [tuple(int(x) - 1 if x else -1 for x in (t.split('/') + ['', ''])[:2]) for t in line[2:].split()]
                    faces.append(idx)
                    mats.append(cur)
    V = np.asarray(vs, dtype=np.float64)
    if not want_faces:
        return V
    return V, np.asarray(vts, dtype=np.float64), faces, mats


def config():
    with open(os.path.join(ICT_DIR, 'vertex_indices.json')) as f:
        return json.load(f)


def load_all():
    """Returns dict(neutral=V, uv=VT, faces=list, mats=list, shapes={name: V})  (cached as npz)."""
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, 'ict.npz')
    names = config()['expressions'] + ['cheekRaiser_L', 'cheekRaiser_R', 'PupilDilate_L', 'PupilDilate_R']
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        d = z['d'].item()
        return d
    V, VT, F, M = read_obj(os.path.join(ICT_DIR, 'generic_neutral_mesh.obj'), True)
    shapes = {}
    for n in names:
        p = os.path.join(ICT_DIR, n + '.obj')
        if os.path.exists(p):
            shapes[n] = read_obj(p)
    d = dict(neutral=V, uv=VT, faces=F, mats=M, shapes=shapes)
    np.savez(cpath, d=np.array(d, dtype=object))
    return d
