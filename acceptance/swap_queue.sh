#!/bin/zsh
# Per-continent swaps of the candidate maps into the Hasterok layout, one run at a time, after the Voronoi baseline
# queue has finished. Output: inversions/results/<stem>.log.
cd "$(dirname $0)/.."
until grep -q "done gcm_glc_vork1500_lcs1_16-133" acceptance/vork_queue.out 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for m in ee2025_poly ee2025_plate nrcan cgmw ecm1 tc1 merdith; do
  c=gcm_glc_swap_${m}_lcs1_16-133
  echo "$(date '+%F %T') start $c"
  nice -n 15 conda run --no-capture-output -n curiesphere python -u -m remit.inversion run inversions/$c.toml \
    > inversions/results/$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c"
  echo "$(date '+%F %T') done $c"
done
