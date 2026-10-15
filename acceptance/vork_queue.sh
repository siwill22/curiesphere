#!/bin/zsh
# Random Voronoi tilings at fixed tile counts (the Eglington & Evans counts 1426 and 1028 first, then a range), one
# run at a time: the score random maps reach for a given number of domains. Output: inversions/results/<stem>.log.
cd "$(dirname $0)/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for k in 1028 250 500 750 1000 1250 1500; do
  c=gcm_glc_vork${k}_lcs1_16-133
  echo "$(date '+%F %T') start $c"
  nice -n 15 conda run --no-capture-output -n curiesphere python -u -m remit.inversion run inversions/$c.toml \
    > inversions/results/$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c"
  echo "$(date '+%F %T') done $c"
done
