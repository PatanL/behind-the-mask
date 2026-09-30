"""Stage 1: build the android head geometry from ICT-FaceKit.

Outputs face/art_raw/cache/mesh.npz with welded skin/teeth/gums geometry, per-vertex
normals, morph deltas (ARKit names), skin weights and procedural eyeballs.
Run: face/build/.venv/bin/python face/build/build_mesh.py
"""
import os, json
import numpy as np
from ict import load_all, REGIONS, SCLERA, CACHE

S = 0.01  # ICT units are centimetres -> metres

# ---------------------------------------------------------------- names
def arkit_name(n):
    if n.endswith('_L'):
        return n[:-2] + 'Left'
    if n.endswith('_R'):
        return n[:-2] + 'Right'
    return n

EXTRA = ['cheekRaiser_L', 'cheekRaiser_R']          # present in FaceXModel but not in the 53-list
COMBINED = {'browInnerUp': ('browInnerUp_L', 'browInnerUp_R'),   # ARKit has one browInnerUp / cheekPuff
            'cheekPuff': ('cheekPuff_L', 'cheekPuff_R')}


def rng(name):
    a, b = REGIONS[name]
    return np.arange(a, b + 1)


def vertex_normals(V, tris):
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])  # area weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, tris[:, k], fn)
    l = np.linalg.norm(N, axis=1, keepdims=True)
    l[l == 0] = 1
    return N / l


def triangulate(faces):
    tv, tt = [], []
    for f in faces:
        for i in range(1, len(f) - 1):
            tv.append((f[0][0], f[i][0], f[i + 1][0]))
            tt.append((f[0][1], f[i][1], f[i + 1][1]))
    return np.array(tv, np.int64), np.array(tt, np.int64)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]
    return c, np.sqrt(x[3] + c @ c)


def main():
    d = load_all()
    V0 = d['neutral'].copy()
    UV = d['uv']
    F, M = d['faces'], d['mats']
    shapes_ict = d['shapes']
    ict_names = [n for n in json.load(open(os.path.join(os.path.dirname(__file__), '..', 'art_raw', 'ict', 'FaceXModel', 'vertex_indices.json')))['expressions']] + EXTRA

    # sanity: ICT "_L" is the subject's left (+x)
    dl = shapes_ict['browDown_L'] - V0
    assert (np.linalg.norm(dl, axis=1) * np.sign(V0[:, 0])).sum() > 0, 'L/R convention'

    # ------------------------------------------------------------ groups of faces
    socket_ids = set(rng('mouthsocket')) | set(rng('eyesocketL')) | set(rng('eyesocketR'))
    groups = {'face': [], 'back': [], 'socket': [], 'teeth': [], 'gums': []}
    for f, m in zip(F, M):
        if m == 'M_Face':
            groups['face'].append(f)
        elif m == 'M_BackHead':
            if any(v in socket_ids for v, _ in f):
                groups['socket'].append(f)
            else:
                groups['back'].append(f)
        elif m == 'M_Teeth':
            groups['teeth'].append(f)
        elif m == 'M_GumsTongue':
            groups['gums'].append(f)

    # ------------------------------------------------------------ deltas (cm)
    deltas = {}
    for n in ict_names:
        deltas[arkit_name(n)] = shapes_ict[n] - V0
    for cn, (a, b) in COMBINED.items():
        deltas[cn] = deltas[arkit_name(a)] + deltas[arkit_name(b)]
    # kill the stray single-vertex noise the ICT shapes carry in the gums
    gum = rng('gums')
    for n, D in deltas.items():
        if not n.startswith('jaw') and n != 'mouthClose':
            D[gum] = 0.0
    morph_names = sorted(deltas.keys(), key=lambda n: ict_order(n, ict_names))

    out = {'morph_names': morph_names, 'prims': {}}
    # ------------------------------------------------------------ welded meshes
    for mesh_name, gnames in [('skin', ['face', 'back', 'socket']), ('teeth', ['teeth']), ('gums', ['gums'])]:
        allf = sum((groups[g] for g in gnames), [])
        tv, _ = triangulate(allf)
        Nn = vertex_normals(V0, tv)
        ndelta = {}
        for n in morph_names:
            if np.abs(deltas[n][np.unique(tv)]).max() < 1e-4:
                continue
            Ns = vertex_normals(V0 + deltas[n], tv)
            ndelta[n] = Ns - Nn
        for g in gnames:
            tvg, ttg = triangulate(groups[g])
            # split vertices by (v, vt) pairs
            pairs = np.stack([tvg.ravel(), ttg.ravel()], 1)
            uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
            vid, tid = uniq[:, 0], uniq[:, 1]
            idx = inv.reshape(-1, 3).astype(np.uint32)
            uv = UV[tid].copy()
            prim = {
                'mesh': mesh_name,
                'orig': vid,
                'pos': (V0[vid] * S).astype(np.float32),
                'nrm': Nn[vid].astype(np.float32),
                'uv': uv.astype(np.float32),
                'idx': idx,
                'morph_pos': {}, 'morph_nrm': {},
            }
            for n in morph_names:
                D = deltas[n][vid] * S
                if np.abs(D).max() < 1e-6:
                    continue
                prim['morph_pos'][n] = D.astype(np.float32)
                prim['morph_nrm'][n] = ndelta.get(n, np.zeros_like(D))[vid].astype(np.float32) if n in ndelta else np.zeros_like(D, dtype=np.float32)
            out['prims'][g] = prim
            print(f'{g:7s} verts={len(vid):6d} tris={len(idx):6d} morphs={len(prim["morph_pos"])}')

    # ------------------------------------------------------------ eyes
    eyes = {}
    for s in 'LR':
        a, b = SCLERA[s]
        P = V0[a:b + 1]
        c, _ = fit_sphere(P)
        ia, ib = {'L': (22221, 23020), 'R': (23791, 24590)}[s]
        ax = (V0[ia:ib + 1] - c).mean(0)
        ax /= np.linalg.norm(ax)
        eyes[s] = {'center': (c * S).tolist(), 'axis': ax.tolist()}
    out['eyes'] = eyes
    out['eye_mesh'] = eyeball_mesh()

    np.savez_compressed(os.path.join(CACHE, 'mesh.npz'), d=np.array(out, dtype=object))
    json.dump({'morph_names': morph_names, 'eyes': eyes}, open(os.path.join(CACHE, 'mesh_info.json'), 'w'), indent=1)
    print('morphs', len(morph_names))


def ict_order(n, ict_names):
    base = [arkit_name(x) for x in ict_names]
    if n in base:
        return base.index(n)
    return 1000 + list(COMBINED).index(n)


def eyeball_mesh(R=1.495, apex=1.672, theta_c=np.radians(32.0), n_phi=72):
    """Surface of revolution around +Z matching the ICT sclera/cornea profile (units cm -> m).
    Returns positions, normals, indices and the cornea parameters used by the eye shader."""
    L = np.array([R * np.sin(theta_c), R * np.cos(theta_c)])
    z0 = (apex ** 2 - L[0] ** 2 - L[1] ** 2) / (2 * (apex - L[1]))
    rc = apex - z0
    # ring angles: dense over the cornea / limbus, coarser at the back
    th = np.concatenate([np.linspace(0, theta_c, 22, endpoint=False),
                         np.linspace(theta_c, np.radians(60), 12, endpoint=False),
                         np.linspace(np.radians(60), np.pi, 20)])
    rings = []
    nrms = []
    for t in th:
        u = np.array([np.sin(t), np.cos(t)])
        if t < theta_c:
            b = z0 * u[1]
            r = b + np.sqrt(b * b - z0 * z0 + rc * rc)
            p = r * u
            n = (p - np.array([0, z0])) / rc
        else:
            r = R
            p = r * u
            n = u
        # soften the crease at the limbus over ~3 degrees
        w = np.clip(1 - abs(t - theta_c) / np.radians(3), 0, 1) * 0.5
        if w > 0:
            nn = (np.array([0, 0]) + u)
            n = n * (1 - w) + nn * w
            n /= np.linalg.norm(n)
        rings.append(p)
        nrms.append(n)
    pos, nrm = [], []
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    for p, n in zip(rings, nrms):
        for ph in phis:
            cph, sph = np.cos(ph), np.sin(ph)
            pos.append((p[0] * cph, p[0] * sph, p[1]))
            nrm.append((n[0] * cph, n[0] * sph, n[1]))
    pos = np.array(pos) * S
    nrm = np.array(nrm)
    idx = []
    nr = len(rings)
    for i in range(nr - 1):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d_ = (i + 1) * n_phi + (j + 1) % n_phi
            if i > 0:
                idx.append((a, c, b))
            idx.append((b, c, d_))
    # (ring 0 and last ring are degenerate poles; harmless zero-area triangles are skipped above for ring 0)
    return {'pos': pos.astype(np.float32), 'nrm': nrm.astype(np.float32), 'idx': np.array(idx, np.uint32),
            'R': R * S, 'apex': apex * S, 'cornea_z0': z0 * S, 'cornea_r': rc * S, 'limbus_z': L[1] * S, 'limbus_r': L[0] * S}


if __name__ == '__main__':
    main()
