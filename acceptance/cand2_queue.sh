#!/bin/zsh
# Batch 2 of the candidate-map test, one run at a time: real data, the S5 synthetic check, then each spin null
# (fewest domains first). Each run's output goes to inversions/results/<stem>.log.
cd "$(dirname $0)/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in gcm_glc_cand2_lcs1_16-133 gcm_glc_cand2_syn_s5 \
         gcm_glc_cand2_null_{nrcan,cgmw}_lcs1_16-133; do
  echo "$(date '+%F %T') start $c"
  nice -n 15 conda run --no-capture-output -n curiesphere python -u -m remit.inversion run inversions/$c.toml \
    > inversions/results/$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c"
  echo "$(date '+%F %T') done $c"
done
