# Deploying ELANG to Hugging Face Spaces (Docker)

The hosted demo runs a lightweight stack — vehicle detection, DeepSORT tracking,
violation heatmap + officer optimizer, E-TLE export, and the citizen-report CRM.
ANPR (PaddleOCR) is left out on the free CPU tier to keep the build small.

## 1. Create the Space

On Hugging Face → **New Space**:

- **SDK:** Docker
- **Hardware:** Free (CPU basic)
- **Visibility:** Public

HF creates an empty Space repo with a `README.md` that carries a YAML header
(`sdk: docker`). **Keep that README** — it is what tells HF how to run the Space.
The Docker SDK serves on port **7860** by default, which matches this `Dockerfile`.

## 2. Push the app into the Space repo (Windows PowerShell)

```powershell
# Clone the Space repo (use the URL HF shows you)
git clone https://huggingface.co/spaces/REGANTENG/<space-name> hf-elang

# Copy the app in, but keep the Space's own README.md and skip heavy/local stuff
robocopy elang-prototype hf-elang /E /XF README.md /XD .git .venv data __pycache__ .cache

# Bring back just the small sample CSVs (handy for reviewers)
New-Item -ItemType Directory -Force hf-elang\data | Out-Null
Copy-Item elang-prototype\data\sample_violations.csv  hf-elang\data\
Copy-Item elang-prototype\data\sample_crm_reports.csv hf-elang\data\

cd hf-elang
git add .
git commit -m "Deploy ELANG demo"
git push
```

HF will build the Docker image and launch the app. First build takes a few minutes
(torch + ultralytics). The first request also downloads YOLOv8-nano (~6 MB) and the
CRM model (~120 MB) into the container cache.

## 3. If you overwrote the Space's README.md

The Space build needs a YAML header at the top of `README.md`. If it went missing,
paste this at the very top:

```yaml
---
title: ELANG Intelligent Traffic Enforcement
emoji: 🦅
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---
```

## Notes

- A login token is required for `git push` to HF — when prompted, use your HF
  username and an access token (Settings → Access Tokens) as the password.
- To enable ANPR on the hosted demo later, add `paddleocr<3.0` and `paddlepaddle<3.0`
  to `requirements-hf.txt` and bump to paid hardware — the free tier struggles with
  the Paddle build.
