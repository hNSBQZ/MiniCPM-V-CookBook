#!/usr/bin/env bash
# ============================================================================
# MiniCPM-o WebRTC Demo One-Click Deployment Script (without Docker)
# Automatically downloads source code, models, compiles inference engine,
# and starts all services.
# Startup order: LiveKit Server -> Backend -> C++ Inference -> Frontend
#
# Simplest usage (one script, fully automatic):
#   PYTHON_CMD=/path/to/python bash one_click.sh start
#
# Full usage:
#   bash one_click.sh {start|stop|restart|status|logs|download}
#
# Environment variable overrides (all have defaults, set as needed):
#   PYTHON_CMD       - Python interpreter path
#   LLAMACPP_ROOT    - llama.cpp-omni source path (auto-downloads if missing)
#   MODEL_DIR        - GGUF model directory (auto-downloads if missing)
#   LLM_QUANT        - LLM quantization: Q4_K_M (default), Q5_K_M, F16, etc.
#   CPP_MODE         - duplex (default) or simplex
#   FRONTEND_MODE    - prod (default) or dev (Vite hot reload)
#   GITHUB_PROXY     - GitHub download proxy
#   HF_ENDPOINT      - HuggingFace mirror (e.g. https://hf-mirror.com)
# ============================================================================

set -euo pipefail

# ======================== Reuse previously installed tools ========================
# Previous install_node / install_livekit installed to ~/.local/bin, ensure they are found on every startup
if [[ -d "$HOME/.local/bin" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# ======================== Path Configuration ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/WebRTC_Demo"
LIVEKIT_CONFIG="$PROJECT_DIR/omini_backend_code/config/livekit.yaml"
BACKEND_DIR="$PROJECT_DIR/omini_backend_code/code"
FRONTEND_DIR="$PROJECT_DIR/o45-frontend"
CPP_SERVER_SCRIPT="$PROJECT_DIR/cpp_server/minicpmo_cpp_http_server.py"
CPP_REF_AUDIO="$PROJECT_DIR/cpp_server/assets/default_ref_audio.wav"

# ======================== Python Configuration (overridable via env vars) ========================
# Specify Python interpreter path (requires >= 3.9, recommended 3.11)
# Usage: PYTHON_CMD=/path/to/python bash one_click.sh start
# PYTHON_CMD="${PYTHON_CMD:-python}"
# pip is auto-derived from the same directory as PYTHON_CMD; can also be overridden separately
# PIP_CMD="${PIP_CMD:-${PYTHON_CMD%/*}/pip}"

# Auto-detect Python: prefer user-specified PYTHON_CMD, then python3, then python
if [[ -z "${PYTHON_CMD:-}" ]]; then
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        PYTHON_CMD="python3"  # let preflight_check report the error
    fi
fi
# Use "python -m pip" instead of standalone pip to guarantee consistency
PIP_CMD="${PIP_CMD:-$PYTHON_CMD -m pip}"


# Bypass SSL certificate verification issues during pip install
PIP_TRUSTED_HOSTS="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org --extra-index-url http://pypi.mirrors.ustc.edu.cn/simple --trusted-host pypi.mirrors.ustc.edu.cn"

# ======================== C++ Inference Configuration (overridable via env vars) ========================
# llama.cpp-omni root directory (requires pre-compiled llama-server)
LLAMACPP_ROOT="${LLAMACPP_ROOT:-$SCRIPT_DIR/llama.cpp-omni}"
# GGUF model directory
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models/openbmb/MiniCPM-o-4_5-gguf}"
# LLM quantization variant (Q4_K_M recommended for most users, F16 for best quality)
# Options: Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0, F16
LLM_QUANT="${LLM_QUANT:-Q4_K_M}"
# Inference mode: duplex (full-duplex) or simplex (half-duplex)
CPP_MODE="${CPP_MODE:-duplex}"
# Vision encoder backend (macOS only): metal (GPU) or coreml (ANE acceleration)
# On Linux, left empty (uses cpp_server default behavior)
VISION_BACKEND="${VISION_BACKEND:-$([ "$(uname -s)" = "Darwin" ] && echo coreml || echo "")}"
# Minimum Node.js version requirement (pnpm 10 requires >= 18.12)
NODE_MIN_VERSION="18"

# ======================== PID Files ========================
PID_DIR="$SCRIPT_DIR/.pids"
LIVEKIT_PID="$PID_DIR/livekit.pid"
BACKEND_PID="$PID_DIR/backend.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"
CPP_SERVER_PID="$PID_DIR/cpp_server.pid"

# ======================== Log Files ========================
LOG_DIR="$SCRIPT_DIR/.logs"
LIVEKIT_LOG="$LOG_DIR/livekit.log"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
CPP_SERVER_LOG="$LOG_DIR/cpp_server.log"

# ======================== LiveKit Configuration ========================
LIVEKIT_API_KEY="devkey"
# v1.9+ requires secret to be at least 32 characters
LIVEKIT_API_SECRET="secretsecretsecretsecretsecretsecret"

# ======================== Source/Model Repository URLs (overridable via env vars) ========================
# Git repository and branch containing WebRTC_Demo
COOKBOOK_REPO="${COOKBOOK_REPO:-https://github.com/OpenSQZ/MiniCPM-V-CookBook.git}"
COOKBOOK_BRANCH="${COOKBOOK_BRANCH:-main}"
# llama.cpp-omni Git repository
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/tc-mb/llama.cpp-omni.git}"
# HuggingFace model repository (GGUF format)
HF_MODEL_REPO="${HF_MODEL_REPO:-openbmb/MiniCPM-o-4_5-gguf}"
# HuggingFace mirror (for faster downloads in China; leave empty for official huggingface.co)
# Reusing the official HF_ENDPOINT environment variable supported by huggingface-cli
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

# ======================== Mirror Acceleration (overridable via env vars) ========================
# GitHub proxy (for faster GitHub Release downloads in China; leave empty for direct connection)
# Common proxies: https://ghfast.top  https://gh-proxy.com  https://mirror.ghproxy.com
GITHUB_PROXY="${GITHUB_PROXY:-https://ghfast.top}"
# Node.js download mirror (leave empty for official nodejs.org)
NODE_MIRROR="${NODE_MIRROR:-https://npmmirror.com/mirrors/node}"
# npm registry mirror
NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmmirror.com}"

# ======================== Ports (overridable via env vars) ========================
# Set AUTO_PORT=1 to automatically find available ports if defaults are in use
AUTO_PORT="${AUTO_PORT:-1}"

# Default ports (will be auto-adjusted if AUTO_PORT=1 and port is in use)
LIVEKIT_PORT="${LIVEKIT_PORT:-7880}"
BACKEND_PORT="${BACKEND_PORT:-8021}"
FRONTEND_PORT="${FRONTEND_PORT:-8088}"
CPP_SERVER_PORT="${CPP_SERVER_PORT:-9060}"
CPP_HEALTH_PORT=$((CPP_SERVER_PORT + 1))
CPP_LLAMA_PORT=$((CPP_SERVER_PORT + 10000))

# Frontend mode: prod=built static server (faster and more stable)  dev=Vite dev server (hot reload)
FRONTEND_MODE="${FRONTEND_MODE:-prod}"

# ======================== Color Output ========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ======================== Utility Functions ========================

get_local_ip() {
    # macOS
    if command -v ipconfig &>/dev/null; then
        local ip
        ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
        if [[ -n "$ip" ]]; then
            echo "$ip"
            return
        fi
    fi
    # Linux / fallback
    if command -v hostname &>/dev/null; then
        hostname -I 2>/dev/null | awk '{print $1}' && return
    fi
    echo "127.0.0.1"
}

check_port() {
    local port=$1
    # Method 1: Direct connection attempt (most reliable, no root needed)
    if (echo >/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
        return 0  # port in use
    fi
    # Method 2: /proc/net/tcp (Linux kernel, no external tools or root needed)
    if [[ -f /proc/net/tcp ]]; then
        local hex_port
        hex_port=$(printf '%04X' "$port")
        if awk -v hp="$hex_port" '$2 ~ ":"hp"$" && $4 == "0A" {found=1; exit} END {exit !found}' /proc/net/tcp 2>/dev/null; then
            return 0  # port in use
        fi
    fi
    # Method 3: lsof (macOS and most Linux, may need sudo for all processes)
    if command -v lsof &>/dev/null && lsof -i :"$port" &>/dev/null; then
        return 0  # port in use
    fi
    # Method 4: ss (modern Linux, -p needs root but -tln works without)
    if command -v ss &>/dev/null && ss -tln 2>/dev/null | grep -q ":${port} "; then
        return 0  # port in use
    fi
    # Method 5: netstat (cross-platform: -an for macOS, -tln for Linux)
    if command -v netstat &>/dev/null; then
        # macOS netstat doesn't support -tln, use -an instead
        if [[ "$(uname)" == "Darwin" ]]; then
            if netstat -an 2>/dev/null | grep -q "\.${port} .*LISTEN"; then
                return 0  # port in use
            fi
        else
            # Linux netstat: -tln works without root
            if netstat -tln 2>/dev/null | grep -q ":${port} "; then
                return 0  # port in use
            fi
        fi
    fi
    return 1  # port free
}

is_port_in_use() {
    # Alias for check_port (for clarity in auto_allocate_ports)
    check_port "$1"
}

# Locate a tool by name, checking PATH and common sbin directories
find_cmd() {
    local cmd=$1
    command -v "$cmd" 2>/dev/null && return 0
    for dir in /usr/sbin /sbin /usr/local/sbin /usr/local/bin; do
        if [[ -x "$dir/$cmd" ]]; then
            echo "$dir/$cmd"
            return 0
        fi
    done
    return 1
}

# Find PIDs listening on a given port (tries multiple tools, no Perl regex)
find_port_pids() {
    local port=$1
    local pids=""

    # Method 1: lsof -ti (macOS and some Linux; also check sbin paths)
    if [[ -z "$pids" ]]; then
        local lsof_cmd
        lsof_cmd=$(find_cmd lsof) && \
            pids=$($lsof_cmd -ti :"$port" 2>/dev/null || true)
    fi
    # Method 2: lsof full output parsing (fallback if -t flag returned empty)
    if [[ -z "$pids" ]]; then
        local lsof_cmd
        lsof_cmd=$(find_cmd lsof) && \
            pids=$($lsof_cmd -i :"$port" 2>/dev/null | awk 'NR>1 {print $2}' | sort -u || true)
    fi
    # Method 3: ss (modern Linux, parse with sed/awk — no grep -oP)
    if [[ -z "$pids" ]] && command -v ss &>/dev/null; then
        pids=$(ss -tlnp 2>/dev/null | grep ":${port} " \
            | sed 's/.*pid=//; s/,.*//' | grep '^[0-9]' || true)
    fi
    # Method 4: netstat (older Linux)
    if [[ -z "$pids" ]] && command -v netstat &>/dev/null; then
        pids=$(netstat -tlnp 2>/dev/null | grep ":${port} " \
            | awk '{print $NF}' | sed 's|/.*||' | grep '^[0-9]' | sort -u || true)
    fi
    # Method 5: fuser (also check sbin paths)
    if [[ -z "$pids" ]]; then
        local fuser_cmd
        fuser_cmd=$(find_cmd fuser) && \
            pids=$($fuser_cmd "${port}/tcp" 2>/dev/null || true)
    fi
    # Method 6: /proc/net/tcp — pure kernel, no external tools needed (Linux only)
    if [[ -z "$pids" ]] && [[ -f /proc/net/tcp ]]; then
        local hex_port
        hex_port=$(printf '%04X' "$port")
        # Find socket inodes in LISTEN state (st=0A) matching the port
        local inodes
        inodes=$(awk -v hp="$hex_port" '$2 ~ ":"hp"$" && $4 == "0A" {print $10}' /proc/net/tcp 2>/dev/null || true)
        for inode in $inodes; do
            [[ -z "$inode" ]] && continue
            # Match inode to process by iterating per-process fd directories
            local pid_num
            for pid_dir in /proc/[0-9]*; do
                pid_num="${pid_dir##*/}"
                # Use ls -l per process (avoids massive glob expansion)
                if ls -l "${pid_dir}/fd" 2>/dev/null | grep -q "socket:\[$inode\]"; then
                    pids="$pids $pid_num"
                    break  # one PID per inode is enough
                fi
            done
        done
    fi

    echo "$pids" | tr -s '[:space:]' '\n' | grep '^[0-9]\+$' | sort -u
}

# Kill the process occupying a given port, then verify the port is freed
ensure_port_free() {
    local port=$1
    local name=${2:-"unknown"}

    if ! check_port "$port"; then
        return 0  # port already free
    fi

    warn "Port $port is occupied, attempting to kill the process for $name..."

    # Try to find PIDs using multiple methods
    local pids
    pids=$(find_port_pids "$port")

    if [[ -n "$pids" ]]; then
        echo "$pids" | while read -r pid; do
            [[ -z "$pid" ]] && continue
            info "Killing process on port $port (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
        done
    else
        # Last resort: try fuser -k directly (it can kill without reporting PIDs)
        local fuser_cmd
        if fuser_cmd=$(find_cmd fuser); then
            info "Using fuser to kill process on port $port..."
            $fuser_cmd -k "${port}/tcp" 2>/dev/null || true
        fi
    fi

    # Wait up to 5s for the port to be freed
    local i=0
    while check_port "$port" && [[ $i -lt 5 ]]; do
        sleep 1
        i=$((i + 1))
    done

    # Force kill if still occupied
    if check_port "$port"; then
        warn "Port $port still occupied after SIGTERM, sending SIGKILL..."
        pids=$(find_port_pids "$port")
        if [[ -n "$pids" ]]; then
            echo "$pids" | while read -r pid; do
                [[ -z "$pid" ]] && continue
                kill -9 "$pid" 2>/dev/null || true
            done
        else
            local fuser_cmd
            if fuser_cmd=$(find_cmd fuser); then
                $fuser_cmd -k -KILL "${port}/tcp" 2>/dev/null || true
            fi
        fi
        sleep 1
    fi

    if check_port "$port"; then
        err "Failed to free port $port. Please kill the process manually:"
        err "  ss -tlnp | grep :${port}    (find PID, then kill <PID>)"
        return 1
    fi

    ok "Port $port freed successfully"
    return 0
}

wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=${3:-15}
    local i=0
    while ! check_port "$port"; do
        sleep 1
        i=$((i + 1))
        if [[ $i -ge $max_wait ]]; then
            err "$name failed to start within ${max_wait}s (port $port)"
            lower_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
            err "Check logs: tail -50 $LOG_DIR/$lower_name.log"
            return 1
        fi
    done
    ok "$name is running (port $port, waited ${i}s)"
}

# Wait for HTTP health check to pass (more reliable than port check, suitable for services with slow model loading)
wait_for_health() {
    local url=$1
    local name=$2
    local max_wait=${3:-180}
    local interval=${4:-5}
    local i=0
    info "Waiting for $name health check to pass (max ${max_wait}s)..."
    while true; do
        local health
        health=$(curl -s "$url" 2>/dev/null || echo "")
        if echo "$health" | grep -q '"healthy"'; then
            ok "$name health check passed (waited ${i}s)"
            return 0
        fi
        sleep "$interval"
        i=$((i + interval))
        if [[ $i -ge $max_wait ]]; then
            err "$name health check timed out (${max_wait}s)"
            err "Check logs: tail -100 $CPP_SERVER_LOG"
            return 1
        fi
        # Print progress periodically
        if (( i % 30 == 0 )); then
            info "Still waiting for $name to start... (${i}s/${max_wait}s)"
        fi
    done
}

is_running() {
    local pidfile=$1
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

find_available_port() {
    local start_port=$1
    local max_attempts=${2:-100}

    for ((i=0; i<max_attempts; i++)); do
        local port=$((start_port + i))
        if ! is_port_in_use "$port"; then
            # Double-check by attempting to bind the port (catches sudo processes)
            if command -v python3 &>/dev/null; then
                if python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', $port)); s.close()" 2>/dev/null; then
                    echo "$port"
                    return 0
                fi
                # Port binding failed (likely in use by sudo process), try next
            else
                # No python3, trust the check_port result
                echo "$port"
                return 0
            fi
        fi
    done

    # If no port found in range, return the start port anyway
    echo "$start_port"
    return 1
}

auto_allocate_ports() {
    # 自动分配可用端口（如果启用 AUTO_PORT=1）
    if [[ "$AUTO_PORT" != "1" ]]; then
        return 0
    fi

    info "Auto-allocating available ports (AUTO_PORT=1)..."

    # LiveKit
    if is_port_in_use "$LIVEKIT_PORT"; then
        local new_port
        new_port=$(find_available_port "$LIVEKIT_PORT")
        if [[ "$new_port" != "$LIVEKIT_PORT" ]]; then
            warn "Port $LIVEKIT_PORT in use, using $new_port for LiveKit"
            LIVEKIT_PORT="$new_port"
        fi
    fi

    # Backend
    if is_port_in_use "$BACKEND_PORT"; then
        local new_port
        new_port=$(find_available_port "$BACKEND_PORT")
        if [[ "$new_port" != "$BACKEND_PORT" ]]; then
            warn "Port $BACKEND_PORT in use, using $new_port for Backend"
            BACKEND_PORT="$new_port"
        fi
    fi

    # Frontend
    if is_port_in_use "$FRONTEND_PORT"; then
        local new_port
        new_port=$(find_available_port "$FRONTEND_PORT")
        if [[ "$new_port" != "$FRONTEND_PORT" ]]; then
            warn "Port $FRONTEND_PORT in use, using $new_port for Frontend"
            FRONTEND_PORT="$new_port"
        fi
    fi

    # C++ Server (also update derived ports)
    if is_port_in_use "$CPP_SERVER_PORT"; then
        local new_port
        new_port=$(find_available_port "$CPP_SERVER_PORT")
        if [[ "$new_port" != "$CPP_SERVER_PORT" ]]; then
            warn "Port $CPP_SERVER_PORT in use, using $new_port for C++ Server"
            CPP_SERVER_PORT="$new_port"
            CPP_HEALTH_PORT=$((CPP_SERVER_PORT + 1))
            CPP_LLAMA_PORT=$((CPP_SERVER_PORT + 10000))
        fi
    fi

    if [[ "$AUTO_PORT" == "1" ]]; then
        ok "Port allocation complete:"
        echo "  LiveKit:     $LIVEKIT_PORT"
        echo "  Backend:     $BACKEND_PORT"
        echo "  Frontend:    $FRONTEND_PORT"
        echo "  C++ Server:  $CPP_SERVER_PORT"
        echo "  C++ Health:  $CPP_HEALTH_PORT"
        echo "  C++ Llama:   $CPP_LLAMA_PORT"
    fi
}

kill_by_pidfile() {
    local pidfile=$1
    local name=$2
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            # Wait for the process to exit
            local i=0
            while kill -0 "$pid" 2>/dev/null && [[ $i -lt 10 ]]; do
                sleep 1
                i=$((i + 1))
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                warn "$name did not respond to SIGTERM, sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            ok "$name stopped"
        else
            info "$name process does not exist (PID: $pid)"
        fi
        rm -f "$pidfile"
    else
        info "$name is not running (no PID file)"
    fi
}

# ======================== Auto-install LiveKit ========================

install_livekit() {
    info "Auto-installing livekit-server..."
    local os_type arch lk_ver lk_os lk_arch dl_url install_dir
    os_type="$(uname -s)"
    arch="$(uname -m)"
    lk_ver="1.9.11"
    install_dir="/usr/local/bin"

    # macOS: prefer Homebrew (no GitHub download needed)
    if [[ "$os_type" == "Darwin" ]] && command -v brew &>/dev/null; then
        info "Installing via Homebrew..."
        if brew install livekit; then
            ok "livekit-server installed successfully: $(livekit-server --version 2>&1 | head -1)"
            return 0
        fi
        warn "Homebrew installation failed, trying direct download..."
    fi

    # Direct binary download (supports GitHub proxy acceleration)
    case "$os_type" in
        Darwin) lk_os="darwin" ;;
        Linux)  lk_os="linux" ;;
        *) err "Unsupported OS: $os_type"; return 1 ;;
    esac
    case "$arch" in
        x86_64)        lk_arch="amd64" ;;
        aarch64|arm64) lk_arch="arm64" ;;
        *) err "Unsupported architecture: $arch"; return 1 ;;
    esac

    local github_url="https://github.com/livekit/livekit/releases/download/v${lk_ver}/livekit_${lk_ver}_${lk_os}_${lk_arch}.tar.gz"
    if [[ -n "$GITHUB_PROXY" ]]; then
        dl_url="${GITHUB_PROXY}/${github_url}"
        info "Downloading via proxy: $dl_url"
    else
        dl_url="$github_url"
        info "Downloading directly: $dl_url"
    fi

    local tmp_dir
    tmp_dir=$(mktemp -d)
    if curl -fSL --progress-bar "$dl_url" | tar -xz -C "$tmp_dir"; then
        # Install to /usr/local/bin (may require sudo)
        if [[ -w "$install_dir" ]]; then
            mv "$tmp_dir/livekit-server" "$install_dir/"
        else
            info "sudo required to install to $install_dir..."
            sudo mv "$tmp_dir/livekit-server" "$install_dir/"
        fi
        chmod +x "$install_dir/livekit-server"
    else
        err "Download failed"
        rm -rf "$tmp_dir"
        return 1
    fi
    rm -rf "$tmp_dir"

    if command -v livekit-server &>/dev/null; then
        ok "livekit-server installed successfully: $(livekit-server --version 2>&1 | head -1)"
    else
        err "livekit-server installation failed, please install manually"
        return 1
    fi
}

# ======================== Auto-install Node.js ========================

install_node() {
    info "Auto-installing Node.js..."
    local os_type arch node_arch node_ver node_os pkg_url install_dir
    os_type="$(uname -s)"
    arch="$(uname -m)"
    node_ver="v22.14.0"  # LTS
    install_dir="$HOME/.local"

    case "$arch" in
        x86_64)        node_arch="x64" ;;
        aarch64|arm64) node_arch="arm64" ;;
        *) err "Unsupported architecture: $arch"; return 1 ;;
    esac

    # macOS: prefer Homebrew (no download needed)
    if [[ "$os_type" == "Darwin" ]] && command -v brew &>/dev/null; then
        info "Installing via Homebrew..."
        if brew install node; then
            ok "Node.js installed successfully: $(node --version 2>&1)"
            return 0
        fi
        warn "Homebrew installation failed, trying direct download..."
    fi

    # Direct download of official binary (supports mirror)
    case "$os_type" in
        Darwin) node_os="darwin"; local ext="tar.gz"; local tar_flag="-xz" ;;
        Linux)  node_os="linux";  local ext="tar.xz"; local tar_flag="-xJ" ;;
        *) err "Unsupported OS: $os_type"; return 1 ;;
    esac

    local filename="node-${node_ver}-${node_os}-${node_arch}.${ext}"
    if [[ -n "$NODE_MIRROR" ]]; then
        pkg_url="${NODE_MIRROR}/${node_ver}/${filename}"
        info "Downloading from mirror: $pkg_url"
    else
        pkg_url="https://nodejs.org/dist/${node_ver}/${filename}"
        info "Downloading: $pkg_url"
    fi

    mkdir -p "$install_dir"
    if curl -fSL --progress-bar "$pkg_url" | tar $tar_flag -C "$install_dir" --strip-components=1; then
        export PATH="$install_dir/bin:$PATH"
    else
        err "Download failed: $pkg_url"
        return 1
    fi

    if command -v node &>/dev/null; then
        ok "Node.js installed successfully: $(node --version 2>&1)"
    else
        err "Node.js installation failed, please install manually: https://nodejs.org/"
        return 1
    fi
}

# ======================== Auto-install pnpm ========================

install_pnpm() {
    info "Auto-installing pnpm..."

    if ! command -v npm &>/dev/null; then
        err "npm is not available, please install Node.js first"
        return 1
    fi

    # Use mirror for faster downloads
    local registry_flag=""
    if [[ -n "$NPM_REGISTRY" ]]; then
        registry_flag="--registry=$NPM_REGISTRY"
        info "Using registry mirror: $NPM_REGISTRY"
    fi

    npm install -g pnpm $registry_flag

    # Verify pnpm can actually run, not just that the binary exists
    if command -v pnpm &>/dev/null && pnpm --version &>/dev/null; then
        ok "pnpm installed successfully: $(pnpm --version 2>&1)"
    else
        err "pnpm installation failed or Node.js version incompatible (requires >= v${NODE_MIN_VERSION})"
        err "Please run manually: npm install -g pnpm"
        return 1
    fi
}

# ======================== Auto-build llama-server ========================

build_llama_server() {
    info "Auto-building llama-server..."
    local os_type cmake_args
    os_type="$(uname -s)"

    # Check build tools
    if ! command -v cmake &>/dev/null; then
        err "cmake is not installed"
        err "  macOS: brew install cmake"
        err "  Linux: apt install cmake or yum install cmake"
        return 1
    fi
    if ! command -v make &>/dev/null && ! command -v ninja &>/dev/null; then
        err "make or ninja is not installed"
        return 1
    fi

    # Base cmake arguments
    cmake_args="-DCMAKE_BUILD_TYPE=Release"

    # Auto-detect GPU acceleration
    case "$os_type" in
        Darwin)
            # macOS: Metal is enabled by default, no extra arguments needed
            info "Detected macOS, using Metal GPU acceleration"
            ;;
        Linux)
            # Linux: detect CUDA
            if command -v nvcc &>/dev/null; then
                local cuda_ver
                cuda_ver=$(nvcc --version 2>&1 | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
                info "Detected CUDA $cuda_ver, enabling CUDA acceleration"
                cmake_args="$cmake_args -DGGML_CUDA=ON"
            elif [[ -d "/usr/local/cuda" ]]; then
                info "Detected CUDA directory /usr/local/cuda, enabling CUDA acceleration"
                cmake_args="$cmake_args -DGGML_CUDA=ON"
                export PATH="/usr/local/cuda/bin:$PATH"
            elif command -v nvidia-smi &>/dev/null; then
                warn "Detected NVIDIA GPU but CUDA toolkit (nvcc) not found"
                warn "Building with CPU only. Install CUDA toolkit for GPU acceleration"
            else
                info "No NVIDIA GPU detected, building with CPU"
            fi
            ;;
    esac

    ok "cmake args: $cmake_args"

    # Execute build
    (
        cd "$LLAMACPP_ROOT"
        info "Configuring (cmake)..."
        cmake -B build $cmake_args 2>&1 | tail -5
        local ncpu
        ncpu=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        info "Building (using ${ncpu} cores, this may take a few minutes)..."
        cmake --build build --target llama-server -j "$ncpu" 2>&1 | \
            while IFS= read -r line; do
                # Show build progress: percentage lines, Linking lines, or errors
                if [[ "$line" =~ ^\[.*%\] ]] || [[ "$line" =~ Linking ]] || [[ "$line" =~ error: ]] || [[ "$line" =~ FAILED ]]; then
                    echo "  $line"
                fi
            done
    ) || {
        err "Build failed, please check the output above"
        return 1
    }

    if [[ -x "$LLAMACPP_ROOT/build/bin/llama-server" ]]; then
        ok "llama-server built successfully: $LLAMACPP_ROOT/build/bin/llama-server"
    else
        err "Build completed but binary not found, please check the build logs"
        return 1
    fi
}

# ======================== Auto-download: WebRTC_Demo ========================

download_webrtc_demo() {
    if [[ -d "$PROJECT_DIR" ]] && [[ -f "$PROJECT_DIR/cpp_server/minicpmo_cpp_http_server.py" ]]; then
        ok "WebRTC_Demo already exists: $PROJECT_DIR"
        return 0
    fi

    info "Auto-downloading WebRTC_Demo source code..."

    if ! command -v git &>/dev/null; then
        err "git is not installed, please install git first"
        return 1
    fi

    # Use GitHub proxy for acceleration (if configured)
    local repo_url="$COOKBOOK_REPO"
    if [[ -n "$GITHUB_PROXY" ]] && [[ "$repo_url" == https://github.com/* ]]; then
        repo_url="${GITHUB_PROXY}/${repo_url}"
        info "Using GitHub proxy: $repo_url"
    fi

    local tmp_dir="$SCRIPT_DIR/.tmp_cookbook_$$"
    trap "rm -rf '$tmp_dir'" EXIT

    # Sparse checkout: only download demo/web_demo/WebRTC_Demo directory
    info "Cloning repository (sparse checkout, downloading WebRTC_Demo only)..."
    git clone --filter=blob:none --no-checkout --depth 1 --branch "$COOKBOOK_BRANCH" \
        "$repo_url" "$tmp_dir" 2>&1 | tail -3 || {
        err "Repository clone failed: $repo_url (branch: $COOKBOOK_BRANCH)"
        rm -rf "$tmp_dir"
        return 1
    }

    (
        cd "$tmp_dir"
        git sparse-checkout set demo/web_demo/WebRTC_Demo
        git checkout 2>&1 | tail -3
    ) || {
        err "Sparse checkout failed"
        rm -rf "$tmp_dir"
        return 1
    }

    # Move WebRTC_Demo to target location
    if [[ -d "$tmp_dir/demo/web_demo/WebRTC_Demo" ]]; then
        mv "$tmp_dir/demo/web_demo/WebRTC_Demo" "$PROJECT_DIR"
        rm -rf "$tmp_dir"
        ok "WebRTC_Demo download complete: $PROJECT_DIR"
    else
        err "WebRTC_Demo directory not found after download, please check the repository structure"
        rm -rf "$tmp_dir"
        return 1
    fi

    trap - EXIT
}

# ======================== Auto-download: llama.cpp-omni ========================

download_llamacpp() {
    if [[ -d "$LLAMACPP_ROOT" ]] && [[ -f "$LLAMACPP_ROOT/CMakeLists.txt" ]]; then
        ok "llama.cpp-omni source already exists: $LLAMACPP_ROOT"
        return 0
    fi

    info "Auto-downloading llama.cpp-omni source code..."

    if ! command -v git &>/dev/null; then
        err "git is not installed, please install git first"
        return 1
    fi

    local repo_url="$LLAMACPP_REPO"
    if [[ -n "$GITHUB_PROXY" ]] && [[ "$repo_url" == https://github.com/* ]]; then
        repo_url="${GITHUB_PROXY}/${repo_url}"
        info "Using GitHub proxy: $repo_url"
    fi

    info "Cloning llama.cpp-omni (this may take a few minutes)..."
    git clone --depth 1 "$repo_url" "$LLAMACPP_ROOT" 2>&1 | tail -5 || {
        err "llama.cpp-omni clone failed: $repo_url"
        return 1
    }

    ok "llama.cpp-omni download complete: $LLAMACPP_ROOT"
}

# ======================== Auto-download: GGUF Models ========================

download_models() {
    if [[ -d "$MODEL_DIR" ]]; then
        # Check if GGUF files exist (not just an empty directory)
        local gguf_count
        gguf_count=$(find "$MODEL_DIR" -name "*.gguf" -maxdepth 1 2>/dev/null | wc -l)
        if [[ "$gguf_count" -gt 0 ]]; then
            ok "Model directory already exists (contains $gguf_count GGUF files): $MODEL_DIR"
            return 0
        fi
    fi

    info "Auto-downloading GGUF models..."

    # Detect or install HuggingFace CLI
    local hf_cli="" hf_dl_cmd=""
    local python_bin_dir
    python_bin_dir="$(dirname "$PYTHON_CMD")"

    # Priority 1: Check if hf/huggingface-cli is now in PATH (including from python env)
    # Temporarily add python bin dir to PATH so conda/venv CLIs are found
    PATH="$python_bin_dir:$PATH"
    if command -v hf &>/dev/null; then
        local hf_path
        hf_path="$(command -v hf)"
        hf_cli="hf"
        hf_dl_cmd="$hf_path download"
        info "Found hf CLI: $hf_path"
    elif command -v huggingface-cli &>/dev/null; then
        local hfcli_path
        hfcli_path="$(command -v huggingface-cli)"
        hf_cli="huggingface-cli"
        hf_dl_cmd="$hfcli_path download"
        info "Found huggingface-cli: $hfcli_path"
    # Priority 2: Try python -m invocation if module is installed
    elif "$PYTHON_CMD" -c "import huggingface_hub" &>/dev/null; then
        hf_cli="python -m huggingface_hub.cli"
        hf_dl_cmd="$PYTHON_CMD -m huggingface_hub.cli download"
    # Priority 3: Not found, attempt to install
    else
        info "huggingface_hub not found, attempting auto-install..."
        $PIP_CMD install -U huggingface_hub $PIP_TRUSTED_HOSTS --progress-bar on || {
            warn "Failed to auto-install huggingface_hub"
        }
        # Re-detect after install
        if command -v hf &>/dev/null; then
            local hf_path
            hf_path="$(command -v hf)"
            hf_cli="hf"
            hf_dl_cmd="$hf_path download"
        elif command -v huggingface-cli &>/dev/null; then
            local hfcli_path
            hfcli_path="$(command -v huggingface-cli)"
            hf_cli="huggingface-cli"
            hf_dl_cmd="$hfcli_path download"
        elif "$PYTHON_CMD" -c "import huggingface_hub" &>/dev/null; then
            hf_cli="python -m huggingface_hub.cli"
            hf_dl_cmd="$PYTHON_CMD -m huggingface_hub.cli download"
        fi
    fi

    if [[ -n "$hf_cli" ]]; then
        info "Using $hf_cli to download model: $HF_MODEL_REPO"
        mkdir -p "$(dirname "$MODEL_DIR")"

        if [[ -n "$HF_ENDPOINT" ]]; then
            info "Using HuggingFace mirror: $HF_ENDPOINT"
        fi

        # Note: hf CLI doesn't support multiple --include flags properly
        # Use Python API for selective download (more reliable)

        info "Downloading selected LLM (${LLM_QUANT}) + all submodels..."
        info "Total size: ~8.9 GB (vs. 79 GB full repo)"

        # Ensure huggingface_hub is installed for Python API
        if ! "$PYTHON_CMD" -c "import huggingface_hub" &>/dev/null; then
            info "Installing huggingface_hub for selective download..."
            $PIP_CMD install -U huggingface_hub $PIP_TRUSTED_HOSTS --progress-bar on || {
                warn "Failed to install huggingface_hub, falling back to CLI download"
                $hf_dl_cmd "$HF_MODEL_REPO" --local-dir "$MODEL_DIR" || {
                    err "Model download failed"
                    return 1
                }
                ok "Model download complete: $MODEL_DIR"
                return 0
            }
        fi

        # Clean up any old temp scripts
        rm -f /tmp/hf_selective_dl_*.py 2>/dev/null || true

        # Create Python script for selective download
        local py_script
        py_script=$(mktemp /tmp/hf_selective_dl_XXXXXX)
        mv "$py_script" "$py_script.py"
        py_script="$py_script.py"

        cat > "$py_script" <<'PYEOF'
import sys
import os
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
llm_quant = sys.argv[3]

# Define patterns for selective download
allow_patterns = [
    f"MiniCPM-o-4_5-{llm_quant}.gguf",  # Selected LLM quantization
    "vision/*",                          # Vision encoder (F16)
    "audio/*",                           # Audio encoder (F16)
    "tts/*",                             # TTS models (F16)
    "token2wav-gguf/*",                  # Token2Wav models
    "*.md",                              # Documentation
    ".git*",                             # Git metadata
]

# Read HF_ENDPOINT from environment (for mirror support)
endpoint = os.environ.get("HF_ENDPOINT", None)

print(f"Downloading with patterns: {allow_patterns}")
print(f"Repo: {repo_id} -> {local_dir}")
if endpoint:
    print(f"Using HuggingFace mirror: {endpoint}")

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
        endpoint=endpoint,  # Support HF_ENDPOINT for mirrors
    )
    print(f"\nDownload complete: {local_dir}")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

        # Run Python script for selective download
        "$PYTHON_CMD" "$py_script" "$HF_MODEL_REPO" "$MODEL_DIR" "$LLM_QUANT" || {
            rm -f "$py_script"
            warn "Selective download failed, falling back to full download..."
            warn "You can manually download only needed files from: ${HF_ENDPOINT:-https://huggingface.co}/${HF_MODEL_REPO}"
            echo ""
            warn "Proceeding with full download in 5 seconds (Ctrl+C to abort)..."
            sleep 5

            $hf_dl_cmd "$HF_MODEL_REPO" --local-dir "$MODEL_DIR" || {
                err "Model download failed"
                return 1
            }
        }
        rm -f "$py_script"

        ok "Model download complete: $MODEL_DIR"
        return 0
    fi

    # Fallback: prompt for manual download
    local base_url="${HF_ENDPOINT:-https://huggingface.co}"
    warn "huggingface-cli not found, please download the model manually:"
    echo ""
    echo "  Option 1: Install huggingface-cli for automatic download"
    echo "    $PIP_CMD install -U huggingface_hub"
    echo "    Then re-run this script"
    echo ""
    echo "  Option 2: Manual download"
    echo "    Visit: $base_url/$HF_MODEL_REPO"
    echo "    Download all files to: $MODEL_DIR"
    echo ""
    echo "  Option 3: Use git lfs"
    echo "    git lfs install"
    echo "    git clone $base_url/$HF_MODEL_REPO $MODEL_DIR"
    echo ""
    return 1
}

# ======================== Preflight Check ========================

preflight_check() {
    info "Running preflight checks..."
    local ok_flag=true

    # ---- Auto-download core source/resources ----
    # WebRTC_Demo (frontend + backend + inference scripts, auto-download if missing)
    if [[ ! -d "$PROJECT_DIR" ]] || [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
        warn "WebRTC_Demo not found, attempting auto-download..."
        download_webrtc_demo || ok_flag=false
    fi

    # llama.cpp-omni (C++ inference engine source, auto-download if missing)
    if [[ ! -d "$LLAMACPP_ROOT" ]] || [[ ! -f "$LLAMACPP_ROOT/CMakeLists.txt" ]]; then
        warn "llama.cpp-omni source not found, attempting auto-download..."
        download_llamacpp || ok_flag=false
    fi

    # GGUF models (attempt auto-download if missing)
    if [[ ! -d "$MODEL_DIR" ]] || [[ $(find "$MODEL_DIR" -name "*.gguf" -maxdepth 1 2>/dev/null | wc -l) -eq 0 ]]; then
        warn "GGUF models not found, attempting auto-download..."
        download_models || ok_flag=false
    fi

    # ---- Tool dependency checks ----

    # livekit-server (auto-install if not found)
    if ! command -v livekit-server &>/dev/null; then
        warn "livekit-server not installed, attempting auto-install..."
        install_livekit || ok_flag=false
    else
        ok "livekit-server $(livekit-server --version 2>&1 | head -1)"
    fi

    # Python (using specified conda environment)
    #if [[ ! -x "$PYTHON_CMD" ]]; then
    #    err "Python not found: $PYTHON_CMD"
    #    err "Please install conda environment: conda create -n py311 python=3.11"
    #    ok_flag=false
    #else
    #    ok "Python: $($PYTHON_CMD --version 2>&1) ($PYTHON_CMD)"
    #fi
    # Python (using specified conda/venv environment)
    if ! command -v "$PYTHON_CMD" &>/dev/null; then
        err "Python not found: $PYTHON_CMD"
        err "Please set PYTHON_CMD=/path/to/python or install Python >= 3.9"
        err "Example: conda create -n py311 python=3.11"
        ok_flag=false
    else
        # Verify Python version >= 3.9
        local py_version
        py_version=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        local py_major="${py_version%%.*}"
        local py_minor="${py_version##*.}"
        if [[ "$py_major" -lt 3 ]] || { [[ "$py_major" -eq 3 ]] && [[ "$py_minor" -lt 9 ]]; }; then
            err "Python >= 3.9 is required, but found $py_version ($PYTHON_CMD)"
            err "Recommended: conda create -n py311 python=3.11"
            ok_flag=false
        else
            # Check if this Python is externally-managed (PEP 668, Ubuntu 23.04+/Debian 12+)
            # If so, auto-create a venv to avoid "externally-managed-environment" error
            local stdlib_path
            stdlib_path=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('stdlib'))" 2>/dev/null || echo "")
            local in_venv=false
            # Already in a venv? (PYTHON_CMD points to a venv or conda env)
            if $PYTHON_CMD -c "import sys; sys.exit(0 if (hasattr(sys, 'real_prefix') or sys.prefix != sys.base_prefix) else 1)" 2>/dev/null; then
                in_venv=true
            fi

            if [[ "$in_venv" == "false" ]] && [[ -n "$stdlib_path" ]] && [[ -f "$stdlib_path/EXTERNALLY-MANAGED" ]]; then
                local venv_dir="$SCRIPT_DIR/.venv"
                if [[ -f "$venv_dir/bin/python" ]]; then
                    info "PEP 668 detected, reusing existing venv: $venv_dir"
                else
                    warn "PEP 668 detected (externally-managed Python), auto-creating venv..."
                    if ! $PYTHON_CMD -m venv "$venv_dir" 2>/dev/null; then
                        err "Failed to create venv. Please install python3-venv:"
                        err "  sudo apt install python3-full python3-venv"
                        ok_flag=false
                    else
                        ok "Virtual environment created: $venv_dir"
                    fi
                fi
                # Switch PYTHON_CMD and PIP_CMD to the venv
                if [[ -f "$venv_dir/bin/python" ]]; then
                    PYTHON_CMD="$venv_dir/bin/python"
                    PIP_CMD="$PYTHON_CMD -m pip"
                    ok "Python (venv): $($PYTHON_CMD --version 2>&1) ($PYTHON_CMD)"
                fi
            else
                ok "Python: $($PYTHON_CMD --version 2>&1) ($PYTHON_CMD)"
            fi
        fi
    fi


    # Node.js (auto-install if not found or version too old)
    local need_node=false
    if ! command -v node &>/dev/null; then
        need_node=true
    else
        local node_major
        node_major=$(node --version 2>&1 | sed 's/v\([0-9]*\).*/\1/')
        if [[ "$node_major" -lt "$NODE_MIN_VERSION" ]]; then
            warn "Node.js $(node --version) is too old (requires >= v${NODE_MIN_VERSION}), auto-upgrading..."
            need_node=true
        fi
    fi
    if [[ "$need_node" == "true" ]]; then
        if ! command -v node &>/dev/null; then
            warn "Node.js not installed, attempting auto-install..."
        fi
        install_node || ok_flag=false
    else
        ok "Node.js: $(node --version 2>&1)"
    fi

    # pnpm (auto-install if not found or unusable)
    local pnpm_ok=false
    if command -v pnpm &>/dev/null; then
        # Verify pnpm can actually run (it may exist but fail if Node version is too low)
        if pnpm --version &>/dev/null; then
            pnpm_ok=true
            ok "pnpm: $(pnpm --version 2>&1)"
        else
            warn "pnpm is installed but cannot run (Node.js version may be incompatible), reinstalling..."
        fi
    else
        warn "pnpm not installed, attempting auto-install..."
    fi
    if [[ "$pnpm_ok" == "false" ]]; then
        install_pnpm || ok_flag=false
    fi

    # ffmpeg (optional dependency for pydub, not required for core functionality)
    if ! command -v ffmpeg &>/dev/null; then
        warn "ffmpeg not installed (optional dependency for pydub, does not affect core functionality). Install: brew install ffmpeg"
    else
        ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
    fi

    # llama-server binary (auto-build if not compiled)
    if [[ -d "$LLAMACPP_ROOT" ]] && [[ -f "$LLAMACPP_ROOT/CMakeLists.txt" ]]; then
        if [[ ! -x "$LLAMACPP_ROOT/build/bin/llama-server" ]]; then
            warn "llama-server not compiled, attempting auto-build..."
            build_llama_server || ok_flag=false
        else
            ok "llama-server: $LLAMACPP_ROOT/build/bin/llama-server"
        fi
    else
        err "llama.cpp-omni source directory missing or incomplete: $LLAMACPP_ROOT"
        ok_flag=false
    fi

    # Model directory
    if [[ -d "$MODEL_DIR" ]] && [[ $(find "$MODEL_DIR" -name "*.gguf" -maxdepth 1 2>/dev/null | wc -l) -gt 0 ]]; then
        ok "Model directory: $MODEL_DIR"
    else
        err "Model directory missing or contains no GGUF files: $MODEL_DIR"
        ok_flag=false
    fi

    # Config files (should all exist after WebRTC_Demo download)
    if [[ ! -f "$LIVEKIT_CONFIG" ]]; then
        err "LiveKit config file not found: $LIVEKIT_CONFIG"
        ok_flag=false
    fi

    if [[ ! -f "$BACKEND_DIR/main.py" ]]; then
        err "Backend entry file not found: $BACKEND_DIR/main.py"
        ok_flag=false
    fi

    if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
        err "Frontend package.json not found: $FRONTEND_DIR/package.json"
        ok_flag=false
    fi

    if [[ ! -f "$CPP_SERVER_SCRIPT" ]]; then
        err "C++ inference server script not found: $CPP_SERVER_SCRIPT"
        ok_flag=false
    fi

    # Check if Docker containers are occupying ports
    if command -v docker &>/dev/null; then
        local docker_containers
        docker_containers=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -E "minicpmo-" || true)
        if [[ -n "$docker_containers" ]]; then
            warn "Detected running Docker containers that may conflict: $docker_containers"
            warn "Consider stopping them first: docker stop $docker_containers"
        fi
    fi

    if [[ "$ok_flag" != "true" ]]; then
        err "Preflight checks failed, please fix the issues above"
        exit 1
    fi

    ok "Preflight checks passed"
}

# ======================== Sync LiveKit Config ========================

# Ensure livekit.yaml keys match the script's LIVEKIT_API_KEY / LIVEKIT_API_SECRET
update_livekit_keys() {
    if [[ ! -f "$LIVEKIT_CONFIG" ]]; then
        return
    fi
    # Replace the line "  devkey: <old_secret>" with the correct secret
    if grep -q "^  ${LIVEKIT_API_KEY}:" "$LIVEKIT_CONFIG"; then
        local current_secret
        current_secret=$(grep "^  ${LIVEKIT_API_KEY}:" "$LIVEKIT_CONFIG" | awk '{print $2}' | tr -d '"' || true)
        if [[ "$current_secret" != "$LIVEKIT_API_SECRET" ]]; then
            sed -i.bak "s/^  ${LIVEKIT_API_KEY}: .*/  ${LIVEKIT_API_KEY}: ${LIVEKIT_API_SECRET}/" "$LIVEKIT_CONFIG"
            rm -f "${LIVEKIT_CONFIG}.bak"
            ok "Updated livekit.yaml API secret (was: ${current_secret:0:6}...)"
        fi
    fi
}

update_livekit_ip() {
    local ip
    ip=$(get_local_ip)
    info "Detected local IP: $ip"

    if grep -q "node_ip:" "$LIVEKIT_CONFIG"; then
        sed -i.bak "s/node_ip: .*/node_ip: \"$ip\"/" "$LIVEKIT_CONFIG"
        sed -i.bak "s/domain: .*/domain: \"$ip\"/" "$LIVEKIT_CONFIG"
        rm -f "${LIVEKIT_CONFIG}.bak"
        ok "Updated livekit.yaml node_ip and domain to $ip"
    fi
}

update_livekit_port() {
    if [[ ! -f "$LIVEKIT_CONFIG" ]]; then
        return
    fi
    # Update port if different from LIVEKIT_PORT
    local current_port
    current_port=$(grep "^port:" "$LIVEKIT_CONFIG" | awk '{print $2}' || echo "7880")
    if [[ "$current_port" != "$LIVEKIT_PORT" ]]; then
        sed -i.bak "s/^port: .*/port: $LIVEKIT_PORT/" "$LIVEKIT_CONFIG"
        rm -f "${LIVEKIT_CONFIG}.bak"
        ok "Updated livekit.yaml port to $LIVEKIT_PORT (was: $current_port)"
    fi
}

# ======================== Start LiveKit ========================

start_livekit() {
    echo ""
    info "========== [1/4] Starting LiveKit Server =========="

    if is_running "$LIVEKIT_PID"; then
        warn "LiveKit is already running (PID: $(cat "$LIVEKIT_PID")), stopping first..."
        kill_by_pidfile "$LIVEKIT_PID" "LiveKit"
    fi

    ensure_port_free "$LIVEKIT_PORT" "LiveKit" || return 1

    update_livekit_keys
    update_livekit_port
    update_livekit_ip

    livekit-server --config "$LIVEKIT_CONFIG" \
        > "$LIVEKIT_LOG" 2>&1 &
    echo $! > "$LIVEKIT_PID"

    wait_for_port "$LIVEKIT_PORT" "LiveKit" 10
}

# ======================== Start Backend ========================

start_backend() {
    echo ""
    info "========== [2/4] Starting Backend (FastAPI) =========="

    if is_running "$BACKEND_PID"; then
        warn "Backend is already running (PID: $(cat "$BACKEND_PID")), stopping first..."
        kill_by_pidfile "$BACKEND_PID" "Backend"
    fi

    ensure_port_free "$BACKEND_PORT" "Backend" || return 1

    # Patch local.yaml to set the correct backend port
    local backend_config="$PROJECT_DIR/omini_backend_code/code/config/local.yaml"
    if [[ -f "$backend_config" ]]; then
        # Check if port line exists and update it
        if grep -q "^  port:" "$backend_config" 2>/dev/null; then
            sed -i.bak "s/^  port:.*/  port: $BACKEND_PORT/" "$backend_config"
            rm -f "${backend_config}.bak"
            ok "Updated backend config port to $BACKEND_PORT"
        else
            # Port line is commented or missing, add it under server:
            sed -i.bak "/^server:/a\\
  port: $BACKEND_PORT" "$backend_config"
            rm -f "${backend_config}.bak"
            ok "Added backend config port: $BACKEND_PORT"
        fi
    else
        warn "Backend config not found: $backend_config (port update skipped)"
    fi

    # Check if backend dependencies are installed (check livekit module, a unique dependency)
    if ! $PYTHON_CMD -c "import livekit" 2>/dev/null; then
        info "Installing backend Python dependencies (first run requires download, please be patient)..."
        (cd "$BACKEND_DIR" && $PIP_CMD install -e . $PIP_TRUSTED_HOSTS --progress-bar on 2>&1) || {
            err "Backend dependency installation failed, please check the output above"
            return 1
        }
        ok "Backend dependencies installed"
    else
        ok "Backend Python dependencies are ready"
    fi

    # Start backend
    (
        cd "$BACKEND_DIR"
        APP_ENV=local \
        SERVER_PORT="$BACKEND_PORT" \
        LIVEKIT_URL="ws://localhost:$LIVEKIT_PORT" \
        LIVEKIT_API_KEY="$LIVEKIT_API_KEY" \
        LIVEKIT_API_SECRET="$LIVEKIT_API_SECRET" \
        WORKERS=1 \
        NUMBA_CACHE_DIR=/tmp/numba_cache \
        $PYTHON_CMD main.py \
            > "$BACKEND_LOG" 2>&1 &
        echo $! > "$BACKEND_PID"
    )

    # Backend startup is slower (VAD model warm-up ~5s), allow 30s timeout
    wait_for_port "$BACKEND_PORT" "Backend" 30

    # Post-startup health check verification
    sleep 2
    local health
    health=$(curl -s http://localhost:$BACKEND_PORT/health 2>/dev/null || echo "FAILED")
    if echo "$health" | grep -q '"healthy"'; then
        ok "Backend health check passed: $health"
    else
        warn "Backend health check not passed (may still be initializing): $health"
    fi
}

# ======================== Start C++ Inference Service ========================

start_cpp_server() {
    echo ""
    info "========== [3/4] Starting C++ Inference Service (model loading takes 2-3 min) =========="

    if is_running "$CPP_SERVER_PID"; then
        warn "C++ inference service is already running (PID: $(cat "$CPP_SERVER_PID")), stopping first..."
        kill_by_pidfile "$CPP_SERVER_PID" "C++ Inference Service"
        pkill -f "minicpmo_cpp_http_server" 2>/dev/null || true
        pkill -f "llama-server.*$CPP_LLAMA_PORT" 2>/dev/null || true
    fi

    ensure_port_free "$CPP_SERVER_PORT" "C++ Inference Service" || return 1

    # Build launch arguments
    local mode_flag="--simplex"
    if [[ "$CPP_MODE" == "duplex" ]]; then
        mode_flag="--duplex"
    fi

    local mode_name="simplex"
    if [[ "$CPP_MODE" == "duplex" ]]; then
        mode_name="duplex"
    fi

    info "Mode: $mode_name | Vision backend: ${VISION_BACKEND:-default} | Port: $CPP_SERVER_PORT"

    # Must run from LLAMACPP_ROOT (llama-server path is relative)
    # Ensure CUDA_VISIBLE_DEVICES is not empty (empty value hides all GPUs)
    local gpu_devices="${CUDA_VISIBLE_DEVICES:-0}"
    (
        cd "$LLAMACPP_ROOT"
        CUDA_VISIBLE_DEVICES="$gpu_devices" \
        REGISTER_URL="http://127.0.0.1:$BACKEND_PORT" \
        REF_AUDIO="$CPP_REF_AUDIO" \
        $PYTHON_CMD "$CPP_SERVER_SCRIPT" \
            --llamacpp-root "$LLAMACPP_ROOT" \
            --model-dir "$MODEL_DIR" \
            --port "$CPP_SERVER_PORT" \
            --gpu-devices "$gpu_devices" \
            ${VISION_BACKEND:+--vision-backend "$VISION_BACKEND"} \
            $mode_flag \
            > "$CPP_SERVER_LOG" 2>&1 &
        echo $! > "$CPP_SERVER_PID"
    )

    # Model loading takes a long time; use the health check port (a lightweight server that starts quickly)
    wait_for_port "$CPP_HEALTH_PORT" "C++ Inference (health check port)" 30

    # Then wait for the main service to become healthy (model loading complete)
    wait_for_health "http://localhost:$CPP_SERVER_PORT/health" "C++ Inference Service" 300 5

    # Force register/re-register inference service (ensure status is available, clear possible stale locks)
    local local_ip
    local_ip=$(get_local_ip)
    local register_result
    register_result=$(curl -s -X POST "http://localhost:$BACKEND_PORT/api/inference/register" \
        -H 'Content-Type: application/json' \
        -d "{\"ip\": \"$local_ip\", \"port\": $CPP_SERVER_PORT, \"model_port\": $CPP_SERVER_PORT, \"model_type\": \"$CPP_MODE\", \"session_type\": \"release\", \"service_name\": \"o45-cpp\"}" 2>/dev/null || echo "")
    if echo "$register_result" | grep -q "service_id"; then
        ok "C++ inference service registered with Backend (status: available)"
    else
        warn "C++ inference service registration failed, manual registration may be needed"
        echo "  curl -X POST http://localhost:$BACKEND_PORT/api/inference/register \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"ip\": \"$local_ip\", \"port\": $CPP_SERVER_PORT, \"model_port\": $CPP_SERVER_PORT, \"model_type\": \"$CPP_MODE\", \"session_type\": \"release\", \"service_name\": \"o45-cpp\"}'"
    fi
}

# ======================== Start Frontend ========================

start_frontend() {
    echo ""
    info "========== [4/4] Starting Frontend (Vue + Vite) =========="

    if is_running "$FRONTEND_PID"; then
        warn "Frontend is already running (PID: $(cat "$FRONTEND_PID")), stopping first..."
        kill_by_pidfile "$FRONTEND_PID" "Frontend"
    fi

    ensure_port_free "$FRONTEND_PORT" "Frontend" || return 1

    # Check node_modules
    if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
        info "Installing frontend dependencies..."
        (cd "$FRONTEND_DIR" && pnpm install ${NPM_REGISTRY:+--registry="$NPM_REGISTRY"}) || {
            err "Frontend dependency installation failed"
            return 1
        }
        ok "Frontend dependencies installed"
    else
        ok "Frontend dependencies are ready (node_modules exists)"
    fi

    # Generate HTTPS self-signed certificate (WebRTC requires secure context, includes server IP)
    local cert_dir="$PROJECT_DIR/.certs"
    local local_ip
    local_ip=$(get_local_ip)
    if [[ ! -f "$cert_dir/server.crt" ]] || ! openssl x509 -in "$cert_dir/server.crt" -noout -text 2>/dev/null | grep -q "$local_ip"; then
        info "Generating HTTPS self-signed certificate (IP: $local_ip)..."
        mkdir -p "$cert_dir"
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "$cert_dir/server.key" \
            -out "$cert_dir/server.crt" \
            -days 365 \
            -subj "/CN=$local_ip" \
            -addext "subjectAltName=IP:$local_ip,IP:127.0.0.1,DNS:localhost" \
            2>/dev/null
        ok "HTTPS certificate generated (IP: $local_ip)"
    else
        ok "HTTPS certificate is ready"
    fi

    # Pass CPP_MODE to frontend so it can gray out unavailable tabs
    info "Frontend will reflect CPP_MODE=$CPP_MODE (unavailable mode tab shown as disabled)"
    export VITE_CPP_MODE="$CPP_MODE"

    if [[ "$FRONTEND_MODE" == "prod" ]]; then
        # Production mode: build first, then serve with static server
        info "Frontend mode: production build (prod)"
        # Check if rebuild is needed (mode changed or dist missing)
        local need_build=false
        local mode_marker="$FRONTEND_DIR/dist/.cpp_mode"
        if [[ ! -d "$FRONTEND_DIR/dist" ]]; then
            need_build=true
        elif [[ "${FORCE_BUILD:-0}" == "1" ]]; then
            need_build=true
        elif [[ ! -f "$mode_marker" ]] || [[ "$(cat "$mode_marker" 2>/dev/null)" != "$CPP_MODE" ]]; then
            info "CPP_MODE changed since last build, rebuilding frontend..."
            need_build=true
        fi
        if [[ "$need_build" == "true" ]]; then
            info "Building frontend..."
            (cd "$FRONTEND_DIR" && VITE_CPP_MODE="$CPP_MODE" pnpm run build:external) || {
                err "Frontend build failed"
                return 1
            }
            echo "$CPP_MODE" > "$mode_marker"
            ok "Frontend build complete"
        else
            ok "Frontend already built (dist/ exists, set FORCE_BUILD=1 to force rebuild)"
        fi
        # Start production HTTPS static server
        (
            cd "$FRONTEND_DIR"
            node serve-prod.mjs \
                --port "$FRONTEND_PORT" \
                --backend "$BACKEND_PORT" \
                --livekit "$LIVEKIT_PORT" \
                > "$FRONTEND_LOG" 2>&1 &
            echo $! > "$FRONTEND_PID"
        )
    else
        # Development mode: Vite dev server
        info "Frontend mode: development server (dev)"
        (
            cd "$FRONTEND_DIR"
            VITE_CPP_MODE="$CPP_MODE" pnpm run dev:external \
                > "$FRONTEND_LOG" 2>&1 &
            echo $! > "$FRONTEND_PID"
        )
    fi

    wait_for_port "$FRONTEND_PORT" "Frontend" 30
}

# ======================== Stop All Services ========================

stop_all() {
    echo ""
    info "Stopping all services..."
    kill_by_pidfile "$FRONTEND_PID"   "Frontend"
    kill_by_pidfile "$CPP_SERVER_PID" "C++ Inference Service"
    # cpp_server forks llama-server child processes, clean them up too
    pkill -f "minicpmo_cpp_http_server" 2>/dev/null || true
    pkill -f "llama-server.*$CPP_LLAMA_PORT" 2>/dev/null || true
    kill_by_pidfile "$BACKEND_PID"    "Backend"
    kill_by_pidfile "$LIVEKIT_PID"    "LiveKit"
    echo ""
    ok "All services stopped"
}

# ======================== Show Status ========================

show_status() {
    echo ""
    info "Service status:"
    echo "  -------------------------------------------------------"

    for svc in livekit backend cpp_server frontend; do
        local pidfile="$PID_DIR/${svc}.pid"
        local port_var
        case $svc in
            livekit)    port_var=$LIVEKIT_PORT ;;
            backend)    port_var=$BACKEND_PORT ;;
            cpp_server) port_var=$CPP_SERVER_PORT ;;
            frontend)   port_var=$FRONTEND_PORT ;;
        esac

        if is_running "$pidfile"; then
            local pid=$(cat "$pidfile")
            echo -e "  ${GREEN}●${NC} $svc\trunning  PID=$pid  port=$port_var"
        else
            echo -e "  ${RED}○${NC} $svc\tnot running  port=$port_var"
        fi
    done

    echo "  -------------------------------------------------------"
    echo ""
    info "Log files:"
    echo "  LiveKit:       $LIVEKIT_LOG"
    echo "  Backend:       $BACKEND_LOG"
    echo "  C++ Inference: $CPP_SERVER_LOG"
    echo "  Frontend:      $FRONTEND_LOG"
    echo ""
    local local_ip
    local_ip=$(get_local_ip)
    info "Access URLs: (frontend mode: $FRONTEND_MODE)"
    echo "  Frontend:      https://$local_ip:$FRONTEND_PORT  (accept self-signed certificate on first visit)"
    echo "  Backend API:   http://$local_ip:$BACKEND_PORT"
    echo "  Backend Docs:  http://$local_ip:$BACKEND_PORT/docs"
    echo "  LiveKit:       ws://$local_ip:$LIVEKIT_PORT"
    echo "  Inference:     http://$local_ip:$CPP_SERVER_PORT"
    echo "  Inference Health: http://$local_ip:$CPP_HEALTH_PORT/health"
}

# ======================== Start All ========================

start_all() {
    echo ""
    echo "=============================================="
    echo "  MiniCPM-o WebRTC Demo One-Click Start (without Docker)"
    echo "=============================================="
    echo ""

    mkdir -p "$PID_DIR" "$LOG_DIR"

    # Auto-allocate available ports if AUTO_PORT=1
    auto_allocate_ports

    preflight_check
    start_livekit
    start_backend
    start_cpp_server
    start_frontend

    echo ""
    echo "=============================================="
    ok "All services started successfully!"
    echo "=============================================="
    show_status
}

# ======================== Main Entry ========================

case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        show_status
        ;;
    logs)
        # Shortcut to view logs: bash one_click.sh logs [livekit|backend|cpp_server|frontend]
        case "${2:-all}" in
            livekit)    tail -f "$LIVEKIT_LOG" ;;
            backend)    tail -f "$BACKEND_LOG" ;;
            cpp|cpp_server) tail -f "$CPP_SERVER_LOG" ;;
            frontend)   tail -f "$FRONTEND_LOG" ;;
            all)        tail -f "$LIVEKIT_LOG" "$BACKEND_LOG" "$CPP_SERVER_LOG" "$FRONTEND_LOG" ;;
        esac
        ;;
    download)
        # Download all dependencies (source + models) only, do not start services
        echo ""
        echo "=============================================="
        echo "  MiniCPM-o WebRTC Demo Dependency Download"
        echo "=============================================="
        echo ""
        dl_ok=true
        download_webrtc_demo || dl_ok=false
        download_llamacpp || dl_ok=false
        download_models || dl_ok=false
        if [[ "$dl_ok" == "true" ]]; then
            echo ""
            ok "All dependencies downloaded! You can now run: bash $0 start"
        else
            echo ""
            err "Some downloads failed, please check the output above"
        fi
        ;;
    update)
        # Pull latest code for WebRTC_Demo (and optionally llama.cpp-omni)
        echo ""
        echo "=============================================="
        echo "  MiniCPM-o WebRTC Demo Code Update"
        echo "=============================================="
        echo ""

        update_ok=true

        # --- Update WebRTC_Demo ---
        # Case 1: oneclick.sh is inside a git repo (e.g. MiniCPM-V-CookBook)
        repo_root=""
        if command -v git &>/dev/null; then
            repo_root=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null || true)
        fi

        if [[ -n "$repo_root" ]] && [[ -d "$repo_root/.git" ]]; then
            info "Detected git repo at: $repo_root"
            info "Pulling latest changes (branch: $(cd "$repo_root" && git branch --show-current 2>/dev/null || echo 'unknown'))..."
            (cd "$repo_root" && git pull --ff-only 2>&1) || {
                warn "git pull --ff-only failed, trying git pull --rebase..."
                (cd "$repo_root" && git pull --rebase 2>&1) || {
                    err "git pull failed. Please resolve conflicts manually in: $repo_root"
                    update_ok=false
                }
            }
            if [[ "$update_ok" == "true" ]]; then
                ok "WebRTC_Demo updated via git pull"
            fi
        # Case 2: PROJECT_DIR was downloaded via sparse checkout (no .git inside)
        elif [[ -d "$PROJECT_DIR" ]] && [[ ! -d "$PROJECT_DIR/.git" ]]; then
            info "Standalone deployment detected, re-downloading WebRTC_Demo..."
            # Backup user config files
            backup_dir="$SCRIPT_DIR/.config_backup_$$"
            mkdir -p "$backup_dir"
            for cfg in "$LIVEKIT_CONFIG" "$BACKEND_DIR/config/local.yaml"; do
                if [[ -f "$cfg" ]]; then
                    cp "$cfg" "$backup_dir/" 2>/dev/null || true
                fi
            done
            info "Config files backed up to: $backup_dir"

            # Remove old and re-download
            rm -rf "$PROJECT_DIR"
            if download_webrtc_demo; then
                # Restore user config files
                for cfg_file in "$backup_dir"/*; do
                    [[ -f "$cfg_file" ]] || continue
                    filename=$(basename "$cfg_file")
                    if [[ "$filename" == "livekit.yaml" ]] && [[ -f "$LIVEKIT_CONFIG" ]]; then
                        cp "$cfg_file" "$LIVEKIT_CONFIG"
                    elif [[ "$filename" == "local.yaml" ]] && [[ -f "$BACKEND_DIR/config/local.yaml" ]]; then
                        cp "$cfg_file" "$BACKEND_DIR/config/local.yaml"
                    fi
                done
                ok "WebRTC_Demo re-downloaded, config files restored"
            else
                err "WebRTC_Demo download failed"
                if [[ -d "$backup_dir" ]]; then
                    warn "Config backup available at: $backup_dir"
                fi
                update_ok=false
            fi
            rm -rf "$backup_dir"
        else
            warn "WebRTC_Demo not found, use 'download' command first"
            update_ok=false
        fi

        # --- Update llama.cpp-omni ---
        if [[ -d "$LLAMACPP_ROOT/.git" ]]; then
            # Has .git — use git pull
            info "Updating llama.cpp-omni (via git pull)..."
            old_commit=$(cd "$LLAMACPP_ROOT" && git rev-parse HEAD 2>/dev/null || echo "")
            (cd "$LLAMACPP_ROOT" && git pull --ff-only 2>&1) && \
                ok "llama.cpp-omni updated" || \
                warn "llama.cpp-omni git pull failed (non-critical)"

            # Auto-rebuild if new commits were pulled
            new_commit=$(cd "$LLAMACPP_ROOT" && git rev-parse HEAD 2>/dev/null || echo "")
            if [[ -n "$old_commit" ]] && [[ "$old_commit" != "$new_commit" ]]; then
                info "New commits detected (${old_commit:0:8} -> ${new_commit:0:8}), rebuilding llama-server..."
                build_llama_server || {
                    err "llama-server rebuild failed"
                    update_ok=false
                }
            elif [[ ! -x "$LLAMACPP_ROOT/build/bin/llama-server" ]]; then
                info "llama-server binary not found, building..."
                build_llama_server || {
                    err "llama-server build failed"
                    update_ok=false
                }
            else
                ok "llama-server binary is up to date, no rebuild needed"
            fi
        elif [[ -d "$LLAMACPP_ROOT" ]]; then
            # Directory exists but no .git — re-download and rebuild
            info "llama.cpp-omni exists but has no .git, re-downloading..."
            rm -rf "$LLAMACPP_ROOT"
            if download_llamacpp; then
                info "Rebuilding llama-server after re-download..."
                build_llama_server || {
                    err "llama-server rebuild failed"
                    update_ok=false
                }
            else
                err "llama.cpp-omni re-download failed"
                update_ok=false
            fi
        else
            info "llama.cpp-omni not found, downloading..."
            if download_llamacpp; then
                build_llama_server || {
                    err "llama-server build failed"
                    update_ok=false
                }
            else
                err "llama.cpp-omni download failed"
                update_ok=false
            fi
        fi

        echo ""
        if [[ "$update_ok" == "true" ]]; then
            ok "Update complete! Run 'bash $0 restart' to apply changes."
        else
            err "Some updates failed, please check the output above"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|download|update}"
        echo ""
        echo "  start     Start all services (auto-downloads missing dependencies)"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  status    Show service status"
        echo "  logs      View logs [livekit|backend|cpp|frontend]"
        echo "  download  Download source code and models only (no start)"
        echo "  update    Pull latest code from git (preserves config files)"
        exit 1
        ;;
esac
