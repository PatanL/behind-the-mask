"""Minimal glTF 2.0 binary (.glb) writer with sparse / quantized morph targets.

Only what the android face needs: meshes with several primitives, morph targets
(sparse accessors, optional normalized int16), a two-joint skin, PNG textures and
PBR materials with KHR_materials_clearcoat.
"""
import json, struct
import numpy as np

FLOAT, UBYTE, USHORT, SHORT, UINT = 5126, 5121, 5123, 5122, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
TYPES = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4', 16: 'MAT4'}
CT = {np.float32: FLOAT, np.uint8: UBYTE, np.uint16: USHORT, np.int16: SHORT, np.uint32: UINT}


class GLB:
    def __init__(self):
        self.j = {'asset': {'version': '2.0', 'generator': 'behind-the-mask face/build'},
                  'scene': 0, 'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [],
                  'accessors': [], 'bufferViews': [], 'buffers': [], 'materials': [],
                  'textures': [], 'images': [], 'samplers': []}
        self.bin = bytearray()
        self.ext_used = set()
        self.ext_req = set()

    # -- raw data --------------------------------------------------------
    def _view(self, data: bytes, target=None, stride=None):
        while len(self.bin) % 4:
            self.bin += b'\0'
        off = len(self.bin)
        self.bin += data
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target:
            v['target'] = target
        if stride:
            v['byteStride'] = stride
        self.j['bufferViews'].append(v)
        return len(self.j['bufferViews']) - 1

    def accessor(self, arr, target=ARRAY_BUFFER, normalized=False, minmax=False):
        arr = np.ascontiguousarray(arr)
        n = arr.shape[0]
        comps = 1 if arr.ndim == 1 else arr.shape[1]
        ct = CT[arr.dtype.type]
        stride = None
        data = arr.tobytes()
        esize = arr.dtype.itemsize * comps
        if target == ARRAY_BUFFER and esize % 4:
            # pad each element to 4 bytes (vertex attribute alignment rule)
            pad = 4 - esize % 4
            raw = np.frombuffer(data, np.uint8).reshape(n, esize)
            raw = np.concatenate([raw, np.zeros((n, pad), np.uint8)], 1)
            data = raw.tobytes()
            stride = esize + pad
        bv = self._view(data, target, stride)
        a = {'bufferView': bv, 'componentType': ct, 'count': int(n), 'type': TYPES[comps]}
        if normalized:
            a['normalized'] = True
        if minmax:
            a['min'] = [float(x) for x in np.atleast_1d(arr.min(0))]
            a['max'] = [float(x) for x in np.atleast_1d(arr.max(0))]
        self.j['accessors'].append(a)
        return len(self.j['accessors']) - 1

    def sparse_accessor(self, dense, eps=0.0, quant=None, minmax=False):
        """dense: (N,3) float deltas. Stores only rows whose |row|>eps.
        quant: None (float32) or scale s -> values stored as int16 normalized (value = q/32767)."""
        dense = np.asarray(dense, np.float32)
        n = dense.shape[0]
        mag = np.abs(dense).max(1)
        idx = np.nonzero(mag > eps)[0]
        a = {'componentType': FLOAT, 'count': int(n), 'type': 'VEC3'}
        if quant:
            a['componentType'] = SHORT
            a['normalized'] = True
        if minmax:
            sub = dense[idx] if len(idx) else np.zeros((1, 3), np.float32)
            a['min'] = [float(min(0.0, x)) for x in sub.min(0)]
            a['max'] = [float(max(0.0, x)) for x in sub.max(0)]
        if len(idx):
            it = np.uint16 if n < 65536 else np.uint32
            iv = self._view(idx.astype(it).tobytes())
            vals = dense[idx]
            if quant:
                vals = np.clip(np.round(vals * 32767.0), -32767, 32767).astype(np.int16)
                if minmax:  # min/max must match the dequantized values
                    deq = vals.astype(np.float64) / 32767.0
                    a['min'] = [float(min(0.0, x)) for x in deq.min(0)]
                    a['max'] = [float(max(0.0, x)) for x in deq.max(0)]
            vv = self._view(vals.tobytes())
            a['sparse'] = {'count': int(len(idx)),
                           'indices': {'bufferView': iv, 'componentType': CT[it]},
                           'values': {'bufferView': vv}}
        self.j['accessors'].append(a)
        return len(self.j['accessors']) - 1, len(idx)

    def image_png(self, png_bytes, name=None, mime='image/png'):
        bv = self._view(png_bytes)
        im = {'bufferView': bv, 'mimeType': mime}
        if name:
            im['name'] = name
        self.j['images'].append(im)
        if not self.j['samplers']:
            self.j['samplers'].append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071})
        self.j['textures'].append({'sampler': 0, 'source': len(self.j['images']) - 1, **({'name': name} if name else {})})
        return len(self.j['textures']) - 1

    def material(self, m):
        self.j['materials'].append(m)
        if 'extensions' in m:
            for k in m['extensions']:
                self.ext_used.add(k)
        return len(self.j['materials']) - 1

    def mesh(self, m):
        self.j['meshes'].append(m)
        return len(self.j['meshes']) - 1

    def node(self, n, root=False):
        self.j['nodes'].append(n)
        i = len(self.j['nodes']) - 1
        if root:
            self.j['scenes'][0]['nodes'].append(i)
        return i

    def write(self, path):
        while len(self.bin) % 4:
            self.bin += b'\0'
        self.j['buffers'] = [{'byteLength': len(self.bin)}]
        for k in ['materials', 'textures', 'images', 'samplers']:
            if not self.j[k]:
                del self.j[k]
        if self.ext_used:
            self.j['extensionsUsed'] = sorted(self.ext_used)
        if self.ext_req:
            self.j['extensionsRequired'] = sorted(self.ext_req)
        js = json.dumps(self.j, separators=(',', ':')).encode()
        while len(js) % 4:
            js += b' '
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack('<II', len(self.bin), 0x004E4942))
            f.write(self.bin)
        return total
