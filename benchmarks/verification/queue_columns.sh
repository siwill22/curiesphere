#!/bin/zsh
# after forward.py: the column cases, one at a time (3 threads, low priority)
cd /Users/simon/GIT/curiesphere
while pgrep -f "verification/forward.py" > /dev/null; do sleep 60; done
export PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 OPENBLAS_NUM_THREADS=1
nice -n 15 ~/anaconda3/envs/curiesphere/bin/python benchmarks/verification/columns.py cells cont_checker cont_geology
