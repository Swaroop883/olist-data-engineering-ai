#!/usr/bin/env bash
# ==============================================================================
# Olist DE + AI Platform — Linux LAN Server Launcher
# Run on your Linux PC so both your PC and your friend's PC (on the same Wi-Fi
# router or mobile hotspot) can use the live application demo simultaneously.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "rag_venv/bin/activate" ]; then
    source "rag_venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

PORT="${1:-8501}"

# Detect local Wi-Fi / LAN IPv4 address
LAN_IP=$(python3 -c '
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
    s.close()
except Exception:
    print("127.0.0.1")
')

echo "========================================================================"
echo " 🚀 OLIST DATA ENGINEERING & AI PLATFORM — LAN SERVER MODE"
echo "========================================================================"
echo " 🖥️  On YOUR Linux PC, open:         http://localhost:${PORT}"
echo " 💻  On your FRIEND'S PC, open:      http://${LAN_IP}:${PORT}"
echo "     (Make sure both PCs are connected to the same Wi-Fi / Hotspot)"
echo ""
echo " 💡  Note: If Ubuntu firewall (ufw) is active and blocks your friend's"
echo "     browser, run once in another terminal:  sudo ufw allow ${PORT}/tcp"
echo "========================================================================"

exec python3 -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT}" \
    --server.fileWatcherType none \
    --server.enableCORS false \
    --server.enableXsrfProtection false
