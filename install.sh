#!/usr/bin/env bash
# ============================================================
# Utopia Client — Linux/macOS installer
# Создаёт venv + зависимости в ~/.utopia-client
# ============================================================
set -e

INSTALL_DIR="${HOME}/.utopia-client"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Utopia Client installer ==="
echo "Папка установки: ${INSTALL_DIR}"
echo

mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# ---- Python ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "[!] python3 не найден. Установи Python 3.10+ и повтори."
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/4] Python ${PY_VER} найден."

# ---- venv ----
if [ ! -d "venv" ]; then
    echo "[2/4] Создаю venv..."
    python3 -m venv venv
else
    echo "[2/4] venv уже существует."
fi

# shellcheck disable=SC1091
source venv/bin/activate

# ---- Зависимости ----
echo "[3/4] Ставлю PyTorch (CPU/GPU — autodetect через системный pip-индекс)..."
pip install --upgrade pip --quiet
pip install torch --quiet
echo "    + основные зависимости (включая neurocore[client] из git)..."
pip install -r "${SCRIPT_DIR}/requirements.txt" numpy --quiet

# ---- Код клиента ----
echo "[4/4] Копирую код клиента..."
rm -rf "${INSTALL_DIR}/utopia_client"
cp -r "${SCRIPT_DIR}/utopia_client" "${INSTALL_DIR}/utopia_client"

# ---- launcher ----
cat > "${INSTALL_DIR}/utopia-client" <<EOF
#!/usr/bin/env bash
cd "${INSTALL_DIR}"
source "${INSTALL_DIR}/venv/bin/activate"
python -m utopia_client.main "\$@"
EOF
chmod +x "${INSTALL_DIR}/utopia-client"

echo
echo "=== Установка завершена ==="
echo "Запуск:"
echo "  ${INSTALL_DIR}/utopia-client benchmark"
echo "  ${INSTALL_DIR}/utopia-client run"
echo
echo "Совет: добавь в PATH: export PATH=\"${INSTALL_DIR}:\$PATH\""
