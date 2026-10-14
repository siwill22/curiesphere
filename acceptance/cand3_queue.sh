#!/bin/zsh
# Batch 3 of the candidate-map test (GLiM, Eglington & Evans 2025 by plate and by polygon, NRCan without ice), one run
# at a time: real data, the four spin nulls, then the 11 S5 draws. Each run's output goes to inversions/results/<stem>.log.
cd "$(dirname $0)/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in gcm_glc_cand3_lcs1_16-133 gcm_glc_cand3_null_{glim,ee2025_plate,ee2025_poly,nrcan}_lcs1_16-133 \
         gcm_glc_cand3_syn_s5_d{0..10}; do
  echo "$(date '+%F %T') start $c"
  nice -n 15 conda run --no-capture-output -n curiesphere python -u -m remit.inversion run inversions/$c.toml \
    > inversions/results/$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c"
  echo "$(date '+%F %T') done $c"
done
