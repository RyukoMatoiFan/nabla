#!/bin/bash
# Nabla Launcher for Linux/macOS
# Run: chmod +x nabla.sh && ./nabla.sh

cd "$(dirname "$0")"

# Check for virtual environment
if [ -f ".venv/bin/python" ]; then
    echo "[+] Using existing virtual environment"
    .venv/bin/python -m nabla
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "[!] Nabla exited with code $exit_code"
        read -p "Press Enter to exit..."
    fi
    exit $exit_code
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found! Please install Python 3.10+"
    exit 1
fi

# Create venv
echo "[+] Creating virtual environment..."
python3 -m venv .venv

# Install dependencies
echo "[+] Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -e . -q

# Launch
echo "[+] Launching Nabla..."
.venv/bin/python -m nabla
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "[!] Nabla exited with code $exit_code"
    read -p "Press Enter to exit..."
fi
