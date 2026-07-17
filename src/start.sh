#!/usr/bin/env bash

# Start SSH server if PUBLIC_KEY is set (enables remote access and dev-sync.sh)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    # Generate host keys if they don't exist (removed during image build for security)
    for key_type in rsa ecdsa ed25519; do
        key_file="/etc/ssh/ssh_host_${key_type}_key"
        if [ ! -f "$key_file" ]; then
            ssh-keygen -t "$key_type" -f "$key_file" -q -N ''
        fi
    done

    service ssh start && echo "worker-comfyui: SSH server started" || echo "worker-comfyui: SSH server could not be started" >&2
fi

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# --- SCAIL volume glue: inputs from network volume + sync finished renders back ---
if [ -d /runpod-volume ]; then
    mkdir -p /runpod-volume/input /runpod-volume/output
    rm -rf /comfyui/input
    ln -s /runpod-volume/input /comfyui/input
    (
      while true; do
        now=$(date +%s)
        for f in /comfyui/output/*.mp4 /comfyui/output/*.png; do
          [ -f "$f" ] || continue
          age=$(( now - $(stat -c %Y "$f") ))
          [ "$age" -gt 10 ] && cp -n "$f" /runpod-volume/output/ 2>/dev/null
        done
        sleep 5
      done
    ) &
    echo "worker-comfyui: SCAIL volume glue active (input symlink + output sync)"
    # boot breadcrumb -> readable from home via S3 (self-diagnosis without logs)
    {
      date
      echo "--- mount ---";   mount | grep runpod-volume
      echo "--- /runpod-volume ---";        ls -la /runpod-volume 2>&1 | head -20
      echo "--- /runpod-volume/input ---";  ls -la /runpod-volume/input 2>&1 | head -20
      echo "--- /comfyui/input (symlink?) ---"; ls -la /comfyui/input 2>&1 | head -20
      echo "--- models seen ---"; ls /runpod-volume/models/diffusion_models /runpod-volume/models/text_encoders/Wan 2>&1
    } > /runpod-volume/boot_debug.txt 2>&1
    # boot patch hook: drop boot_patch.sh on the volume -> runs every boot (no image rebuild needed)
    [ -f /runpod-volume/boot_patch.sh ] && bash /runpod-volume/boot_patch.sh || true
fi


# ---------------------------------------------------------------------------
# GPU pre-flight check
# Verify that the GPU is accessible before starting ComfyUI. If PyTorch
# cannot initialize CUDA the worker will never be able to process jobs,
# so we fail fast with an actionable error message.
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # Launch a real kernel. The driver-only calls above succeed even when this
    # PyTorch build has no compiled kernels for the GPU architecture (e.g. an
    # older torch on a newer GPU). Without this, the worker boots, ComfyUI dies
    # on the first GPU op, and it surfaces as the misleading 'server not
    # reachable' error instead of a clear cause here.
    _ = (torch.zeros(8, device='cuda') + 1).sum().item()
    torch.cuda.synchronize()
    print(f'OK: {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}, cuda {torch.version.cuda}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-comfyui: GPU is not available or incompatible with this PyTorch build:"
    echo "worker-comfyui: $GPU_CHECK"
    echo "worker-comfyui: A 'no kernel image is available' error means this torch build"
    echo "worker-comfyui: lacks kernels for this GPU. Otherwise the GPU may not be"
    echo "worker-comfyui: properly initialized — please contact RunPod support."
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# PID file used by the handler to detect if ComfyUI is still running
COMFY_PID_FILE="/tmp/comfyui.pid"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi