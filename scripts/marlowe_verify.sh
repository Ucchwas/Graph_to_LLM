#!/bin/bash
# Marlowe first-login verification (runs on a LOGIN node; read-only, no jobs).
# Resolves the unknowns in docs/hpc/marlowe.md section 10.
set -u
echo "=== who/where ==="; hostname; id; date
echo "=== slurm associations (is marlowe-m000211-pm06 real? bare account too?) ==="
sacctmgr show assoc user=$USER format=account%30,partition,qos%20 2>&1
echo "=== partitions (current limits -- trust this, not the docs) ==="
scontrol show partition | grep -E "PartitionName|MaxTime|MaxNodes|State|AllowAccounts" 
echo "=== array/job caps ==="
scontrol show config | grep -iE "maxarraysize|maxjobcount" 
echo "=== storage ==="
ls -d /projects/m000211 2>&1; df -h /projects/m000211 /scratch/m000211 /scratch/m000211-pm06 ~ 2>&1
quota -s 2>/dev/null; lfs quota -h -p $(stat -c %g /scratch/m000211 2>/dev/null) /scratch/m000211 2>/dev/null
echo "=== modules ==="
module avail 2>&1 | head -40
echo "=== python on login node ==="
which python3; python3 --version 2>&1
echo "=== gpu driver (from a node view) ==="
sinfo -N -o "%N %G %f" | head -5
scontrol show node n01 2>/dev/null | grep -E "Gres|TmpDisk|AvailableFeatures"
echo "=== internet probe (login node) ==="
curl -sSI --max-time 5 https://huggingface.co | head -1
curl -sSI --max-time 5 https://github.com | head -1
echo "=== done ==="
