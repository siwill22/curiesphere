"""
Content-addressed cache (ADR 0004): each expensive result is stored under the hash of exactly the inputs that
determine it, in a directory shared by all inversions, outside the repository. The root is
$CURIESPHERE_CACHE, default ~/.cache/curiesphere. Each entry is a directory <kind>/<hash>/ holding spec.json
(the inputs, for inspection) and the result files. Each kind's spec carries the version of the algorithm that computes it, so a fix never reuses old results.
Files are written atomically, so an interrupted run never
leaves a half-written entry; deleting any part of the cache is always safe.
"""
import json
import os

import numpy as np

from .config import spec_hash


class Cache:
    def __init__(self, root=None):
        self.root = os.path.expanduser(root or os.environ.get('CURIESPHERE_CACHE', '~/.cache/curiesphere'))

    def entry(self, kind, spec):
        """directory of the entry for (kind, spec), created with its spec.json if new"""
        d = os.path.join(self.root, kind, spec_hash(spec))
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, 'spec.json')
        if not os.path.exists(f):
            save_atomic(f, lambda p: open(p, 'w').write(json.dumps(spec, indent=1, sort_keys=True, default=str)))
        return d

    def array(self, kind, spec, name, compute, mmap=False):
        """the array <name>.npy of the entry, computed (and stored) if missing"""
        f = os.path.join(self.entry(kind, spec), f'{name}.npy')
        if not os.path.exists(f):
            a = compute()
            save_atomic(f, lambda p: np.save(p, a))
        return np.load(f, mmap_mode='r' if mmap else None)


def save_atomic(path, write):
    """write(tmp) to a temporary name next to path, then rename it into place"""
    tmp = f'{path}.tmp{os.getpid()}'
    if path.endswith('.npy'):
        tmp += '.npy'                                    # np.save appends .npy otherwise
    write(tmp)
    os.replace(tmp, path)
