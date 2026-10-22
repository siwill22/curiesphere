"""Stage 0b inputs (pygmt17): the whitened, area-scaled GCM-GLC matrix B and data, assembled as in run_global_continents.py"""
import os, sys, time, numpy as np
src = open('run_global_continents.py').read()
head = src.split('# 5. eigendecomposition')[0]
sys.argv = ['x', 'regions/global_continents/config.toml']
t = time.time(); exec(head); print(f'B assembled in {time.time()-t:.0f} s', B.shape)
OUT = os.environ.get('STAGE0_DIR', os.path.expanduser('~/.cache/curiesphere/stage0'))
os.makedirs(OUT, exist_ok=True)
np.save(f'{OUT}/B.npy', B)
r = np.load('regions/global_continents/cache/lcs1_16-133/results.npz')
np.savez(f'{OUT}/gcm.npz', y=w*g_data, u=u, a=a, alpha2=r['real_evid_alpha2'], m=r['real_evid_m'], sd=r['real_evid_sd'],
         v0=r['real_evid_v0'], sigma=r['real_evid_sigma'])
