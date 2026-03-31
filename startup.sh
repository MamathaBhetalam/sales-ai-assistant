#!/usr/bin/env bash
# Azure App Service (Linux) — bind Streamlit to the platform port.
set -euo pipefail
PORT="${PORT:-8000}"
exec python -m streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true
