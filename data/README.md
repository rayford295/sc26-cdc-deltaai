# Data Setup on DeltaAI

The dataset (100_0005, ~3.18 GB) and model weights are **not** stored in this repository.
Upload them manually to DeltaAI before running jobs.

## 1. Upload data from your local machine

```bash
# From your local machine (Mac terminal)
scp -r /path/to/100_0005 YOUR_NETID@dtai-login.delta.ncsa.illinois.edu:/projects/YOUR_PROJECT/data/
```

Or use rsync (faster for large files, resumable):
```bash
rsync -avP /path/to/100_0005 YOUR_NETID@dtai-login.delta.ncsa.illinois.edu:/projects/YOUR_PROJECT/data/
```

## 2. Download model weights on DeltaAI

SSH into DeltaAI, then:
```bash
mkdir -p /projects/YOUR_PROJECT/weights
cd /projects/YOUR_PROJECT/weights

# Install huggingface_hub if needed
pip install huggingface_hub

# Download weights
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='rhyang/CDC_params', filename='epsilon_lpips0.9.pt', local_dir='.')
hf_hub_download(repo_id='rhyang/CDC_params', filename='epsilon_lpips0.0.pt', local_dir='.')
"
```

## 3. Update paths in scripts/run_deltaai.sh

Edit the following lines with your actual paths:
- `--account=YOUR_ALLOCATION`
- `DATA_DIR=/projects/YOUR_PROJECT/data/100_0005`
- `CKPT_DIR=/projects/YOUR_PROJECT/weights`
