"""Inspect WiSig SingleDay.pkl structure."""
import pickle
import numpy as np

with open('data/SingleDay.pkl', 'rb') as f:
    ds = pickle.load(f)

print('top-level type:', type(ds))
if isinstance(ds, dict):
    for k, v in ds.items():
        print(f'  key={k!r:30s} type={type(v).__name__}', end='')
        if isinstance(v, (list, tuple)):
            print(f' len={len(v)}', end='')
            if len(v) > 0:
                print(f' first={type(v[0]).__name__}', end='')
                if isinstance(v[0], np.ndarray):
                    print(f' shape={v[0].shape} dtype={v[0].dtype}', end='')
        elif isinstance(v, np.ndarray):
            print(f' shape={v.shape} dtype={v.dtype}', end='')
        print()

if isinstance(ds, dict) and 'data' in ds:
    d = ds['data']
    print('\ndata structure detail:')
    print('  outer len (#tx?):', len(d))
    print('  d[0] type:', type(d[0]).__name__)
    if isinstance(d[0], (list, tuple)):
        print('  d[0] len:', len(d[0]))
        if len(d[0]) > 0:
            print('  d[0][0] type:', type(d[0][0]).__name__)
            if isinstance(d[0][0], (list, tuple)):
                print('  d[0][0] len:', len(d[0][0]))
                print('  d[0][0][0] type:', type(d[0][0][0]).__name__)
                if isinstance(d[0][0][0], (list, tuple)):
                    print('  d[0][0][0] len:', len(d[0][0][0]))
                    inner = d[0][0][0][0]
                    print('  d[0][0][0][0] type:', type(inner).__name__,
                          'shape:', getattr(inner, 'shape', '?'),
                          'dtype:', getattr(inner, 'dtype', '?'))
    elif isinstance(d[0], np.ndarray):
        print('  d[0] shape:', d[0].shape, 'dtype:', d[0].dtype)
