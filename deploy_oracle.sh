#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <VM_PUBLIC_IP> <PATH_TO_SSH_KEY> [SSH_USER=ubuntu]"
  exit 1
fi

VM_IP="$1"
SSH_KEY="$2"
SSH_USER="${3:-ubuntu}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="~/shorts-studio"

if [[ ! -f "$SSH_KEY" ]]; then
  echo "Error: SSH key not found at $SSH_KEY"
  exit 1
fi

chmod 600 "$SSH_KEY"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

echo "==> 1/4 Installing Docker & opening firewall port 7860 on $VM_IP..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${VM_IP}" bash -s <<'EOF'
set -euo pipefail
sudo apt-get update -qq
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get install -y -qq docker.io
fi
if ! docker compose version >/dev/null 2>&1; then
  sudo apt-get install -y -qq docker-compose-v2 || sudo apt-get install -y -qq docker-compose
fi
sudo usermod -aG docker "$USER" || true
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow 22/tcp >/dev/null || true
  sudo ufw allow 7860/tcp >/dev/null || true
  sudo ufw --force enable >/dev/null || true
fi
mkdir -p ~/shorts-studio/data/jobs ~/shorts-studio/outputs
EOF

echo "==> 2/4 Syncing project code (excluding .venv, work/, outputs/, GitHub mirror)..."
rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='work/' \
  --exclude='uploads/' \
  --exclude='outputs/' \
  --exclude='data/jobs/' \
  --exclude='shorts-studio-github/' \
  --exclude='cookies_*.txt' \
  --exclude='.DS_Store' \
  "${PROJECT_DIR}/" "${SSH_USER}@${VM_IP}:~/shorts-studio/"

echo "==> 3/4 Building and starting 24/7 container..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${VM_IP}" bash -s <<'EOF'
set -euo pipefail
cd ~/shorts-studio
mkdir -p data/jobs outputs
if [[ -f data/settings.json ]]; then
  python3 -c '
import json
from pathlib import Path
p = Path("data/settings.json")
d = json.loads(p.read_text(encoding="utf-8"))
if str(d.get("storage_dir", "")).startswith("/Users/"):
    d["storage_dir"] = "/home/user/app"
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
'
fi
if sudo docker compose version >/dev/null 2>&1; then
  sudo docker compose up -d --build
else
  sudo docker-compose up -d --build
fi
EOF

echo "==> 4/4 Checking Studio health..."
sleep 5
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${VM_IP}" "curl -fsS http://127.0.0.1:7860/api/health"
echo ""
echo "Deployed! Open http://${VM_IP}:7860 and http://${VM_IP}:7860/monitor"
