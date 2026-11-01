# Deploying CopulaDeriv Studio on Streamlit Community Cloud

CopulaDeriv Studio runs on the **free** Streamlit Community Cloud tier. No paid
services, no API keys, no database and no local machine are involved.

## 1. Repository preparation

The application directory is `copuladeriv/` and contains everything the cloud
runtime needs:

```text
copuladeriv/
    app.py                  # Streamlit application (main file to select)
    studio_core.py          # experiment controller layer (imports app-side)
    requirements.txt        # app-specific dependencies (streamlit, plotly, pandas, numpy, scipy)
    .streamlit/config.toml  # dark research-terminal theme
    python/copuladeriv/     # the validated pricing package (imported via sys.path bootstrap)
```

`studio_core.py` adds `<app_dir>/python` to `sys.path` at import time, so a fresh
clone works without a repository-wide installation step. No developer-machine
paths are referenced.

Push the repository to GitHub (Sparkydev007/CopulaDeriv or your fork).

## 2. Deployment steps

1. Sign in at <https://share.streamlit.io> with your GitHub account.
2. Click **Create app** (or *New app*).
3. **Repository:** select `Sparkydev007/CopulaDeriv`.
4. **Branch:** `main` (or the branch you pushed).
5. **Main file path:** `copuladeriv/app.py` — note the subfolder; the root
   `app.py` does not exist, the file lives inside `copuladeriv/`.
6. **App URL:** choose a subdomain such as `copuladeriv-studio`.
7. Click **Deploy**. The first build installs `copuladeriv/requirements.txt`.
8. Expect the first load of an experiment page to be slower than local use; the
   free tier runs on shared CPU resources.

## 3. What the deployed app can and cannot do

| Capability | Status on Cloud |
|---|---|
| Pricing Laboratory, Copula Comparison, Sensitivity, Tail Dependence, Convergence | ✅ full functionality (path counts capped at 500 000) |
| Experiment History | ⚠️ file-based storage (`research/dashboard_experiments/experiments.jsonl`) — **not durable**: an app restart or redeploys can wipe it. Use the export (JSON/CSV) buttons to keep results locally. |
| C++20 benchmark execution | ❌ native binaries cannot run. The C++ page explains and links to the local workflow below. |
| Live market data / historical calibration | ❌ intentionally out of scope; the app never reuses physical-measure estimates. |

## 4. Local C++20 benchmark workflow (not cloud-deployable)

```bash
cd copuladeriv/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/copuladeriv          # or build/copuladeriv.exe on Windows
```

When you publish C++ numbers, document compiler, flags, CPU and OS, use the same
configuration and path counts on both sides, and compare **end-to-end with
end-to-end** (or core compute with core compute). Monte Carlo throughput
(paths/second) is not HFT event-processing latency and must not be advertised as
such.

## 5. Updating the deployed app

Commit changes to the selected branch and push. Streamlit Community Cloud picks
up the change automatically (usually within a minute). Use **Reboot app** in the
cloud menu if a manual restart is needed — remember this also clears the
file-based experiment history.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on missing module | Check `copuladeriv/requirements.txt` lists the import (streamlit, plotly, pandas, numpy, scipy). |
| `ModuleNotFoundError: copuladeriv` | The main file path must be `copuladeriv/app.py` (with `copuladeriv/` prefix), not a root-level file. |
| App sleeps or `This app has gone to sleep` | Free-tier apps sleep after inactivity; press the wake button in the browser or the dashboard. |
| Experiment history is empty | Storage is ephemeral on Cloud. Re-run, or keep exported JSON/CSV locally. |
| Out-of-memory on very large path counts | Reduce N (≤ 200 000 recommended on the free tier) or raise artificial limits only locally. |
| Slow first experiment | Shared-instance cold start; subsequent identical runs are served from the Streamlit cache keyed by the exact configuration. |

## 7. Secrets

None are required for normal operation. Do not add any credentials, tokens or
private keys to the repository for this deployment.
