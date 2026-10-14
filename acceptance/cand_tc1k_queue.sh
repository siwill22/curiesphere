#!/bin/zsh
# TC1 with the Keller grid: real data, its 20 spun copies (the cand1 null config, rerun) and S5 draw 0, one at a time,
# after the S5 draw queue has finished. Each run's output goes to inversions/results/<stem>.log.
cd "$(dirname $0)/.."
until grep -q "done gcm_glc_cand_syn_s5_d10" acceptance/cand_s5_queue.out 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in gcm_glc_cand_tc1k_lcs1_16-133 gcm_glc_cand_tc1k_syn_s5 gcm_glc_cand1_null_tc1_lcs1_16-133; do
  echo "$(date '+%F %T') start $c"
  nice -n 15 conda run --no-capture-output -n curiesphere python -u -m remit.inversion run inversions/$c.toml \
    > inversions/results/$c.log 2>&1 || echo "$(date '+%F %T') FAILED $c"
  echo "$(date '+%F %T') done $c"
done
