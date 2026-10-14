#!/bin/zsh
# Ten more S5 draws for the candidate-map test (seeds of draw i: truth 20261010+i, noise 20261007+i), one at a time
cd "$(dirname $0)/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in gcm_glc_cand_syn_s5_d{1..10}; do
  echo "$(date '+%F %T') start $c"
  nice -n 15 conda run --no-capture-output -n curiesphere python -u -m remit.inversion run inversions/$c.toml \
    > inversions/results/$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c"
  echo "$(date '+%F %T') done $c"
done
