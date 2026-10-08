"""Stage 0b: BB' (the GCM-GLC data-space Gram matrix, 17700^2 from 17700 x 22591), its eigendecomposition, and the
effect of forming it in float32 on the GPU on the evidence solution (Cholesky at the stored alpha^2, v0 profiled).
Env: LABEL, EIGH=1 to time the eigendecomposition, MPS=1 to run the GPU part (needs torch)."""
import os, time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

D = os.environ.get('STAGE0_DIR', os.path.expanduser('~/.cache/curiesphere/stage0'))
label = os.environ.get('LABEL', '')
B = np.load(f'{D}/B.npy')
z = np.load(f'{D}/gcm.npz')
print(f'== {label}: numpy {np.__version__}', flush=True)

t = time.time(); C64 = B @ B.T; t64 = time.time() - t
print(f"BB' float64 CPU: {t64:.1f} s", flush=True)


def solve(C):
    """GCM evidence solution at the stored alpha^2: x = B'(BB' + a2 I)^-1 (y - v0 u), v0 = u'C^-1 y / u'C^-1 u"""
    a2 = float(z['alpha2'])
    f = cho_factor(C + a2*np.eye(len(C)))
    Ciy, Ciu = cho_solve(f, z['y']), cho_solve(f, z['u'])
    v0 = (z['u'] @ Ciy)/(z['u'] @ Ciu)
    x = B.T @ (Ciy - v0*Ciu)
    return v0 + x/np.sqrt(z['a']), v0


t = time.time(); m64, v064 = solve(C64); tc = time.time() - t
sd = z['sd']
print(f'Cholesky solve: {tc:.1f} s; v0 {v064:.4f} (stored {float(z["v0"]):.4f}); '
      f'max |m - stored|/sd {np.max(np.abs(m64 - z["m"])/sd):.1e}', flush=True)

if os.environ.get('MPS') == '1':
    import torch
    Bt = torch.as_tensor(B, dtype=torch.float32, device='mps'); torch.mps.synchronize()
    (Bt[:100] @ Bt[:100].T).cpu()                                     # warm-up
    t = time.time(); C32 = (Bt @ Bt.T); torch.mps.synchronize(); C32 = C32.cpu().double().numpy(); t32 = time.time() - t
    m32, v032 = solve(C32)
    d = np.abs(m32 - m64)/sd
    print(f"BB' float32 MPS: {t32:.1f} s (incl. copy back); max rel diff of BB' {np.abs(C32 - C64).max()/np.abs(C64).max():.1e}; "
          f'v0 {v032:.5f} vs {v064:.5f}; |m32 - m64|/sd: median {np.median(d):.1e}, max {d.max():.1e}', flush=True)

if os.environ.get('EIGH') == '1':
    t = time.time(); lam, U = eigh(C64, overwrite_a=False, check_finite=False); te = time.time() - t
    print(f'eigh float64 CPU ({len(C64)}^2): {te/60:.1f} min', flush=True)
