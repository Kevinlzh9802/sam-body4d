#!/bin/bash
# Standalone GPU ECC check. Uses none of the project's code.
# Submit on the node that failed:
#   sbatch --nodelist=<node> ecc_repro.sbatch
# Optional: SIF=/path/to/image.sif sbatch ... runs the torch test inside that image
# (otherwise it uses whatever python3 has torch on the node).
#SBATCH --job-name=ecc_repro
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=all
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=ecc_repro_%j.out

echo "host=$(hostname)  job=$SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  date=$(date -Is)"

echo "== 1. Driver view (no CUDA context) =="
nvidia-smi --query-gpu=index,name,pci.bus_id,serial,uuid,driver_version,ecc.mode.current,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv
nvidia-smi -q -d ECC,ROW_REMAPPER,PAGE_RETIREMENT

echo "== 2. Allocate, write and read back ~90% of GPU memory =="
cat > /tmp/ecc_memtest_$SLURM_JOB_ID.py <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
torch.zeros(1, device="cuda"); torch.cuda.synchronize()          # context creation
print("device:", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
chunk = 1 << 30                                                   # 1 GiB
n = int(free * 0.9) // chunk
print(f"free={free/2**30:.1f} GiB total={total/2**30:.1f} GiB, testing {n} x 1 GiB")
bufs = []
for i in range(n):
    b = torch.empty(chunk // 4, dtype=torch.int32, device="cuda")
    b.fill_(0x5A5A5A5A ^ i)
    bufs.append(b)
torch.cuda.synchronize()
for i, b in enumerate(bufs):
    bad = (b != (0x5A5A5A5A ^ i)).sum().item()
    if bad:
        print(f"chunk {i}: {bad} mismatched words")
x = torch.randn(8192, 8192, device="cuda")
for _ in range(20):
    x = (x @ x).clamp_(-1, 1)
torch.cuda.synchronize()
print("PASS: no CUDA error")
EOF
if [ -n "${SIF:-}" ]; then
  apptainer exec --nv -B /tmp "$SIF" python /tmp/ecc_memtest_$SLURM_JOB_ID.py
else
  python3 /tmp/ecc_memtest_$SLURM_JOB_ID.py
fi
echo "exit code: $?"

echo "== 3. Driver view after the test =="
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv
nvidia-smi -q -d ROW_REMAPPER
rm -f /tmp/ecc_memtest_$SLURM_JOB_ID.py
