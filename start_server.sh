#!/bin/bash

set -e -o pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

SERVER_DIR="$WORKSPACE_DIR/vast-pyworker"
ENV_PATH="$WORKSPACE_DIR/worker-env"
DEBUG_LOG="$WORKSPACE_DIR/debug.log"
PYWORKER_LOG="$WORKSPACE_DIR/pyworker.log"

REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
USE_SSL="${USE_SSL:-true}"
WORKER_PORT="${WORKER_PORT:-3000}"

mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# make all output go to $DEBUG_LOG and stdout without having to add `... | tee -a $DEBUG_LOG` to every command
exec &> >(tee -a "$DEBUG_LOG")

function echo_var(){
    echo "$1: ${!1}"
}

[ -z "$BACKEND" ] && echo "BACKEND must be set!" && exit 1
[ -z "$MODEL_LOG" ] && echo "MODEL_LOG must be set!" && exit 1
[ -z "$HF_TOKEN" ] && echo "HF_TOKEN must be set!" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID must be set!" && exit 1

[ -z "$MODEL_VERSION" ] && MODEL_VERSION="main"


echo "start_server.sh"
date

echo_var BACKEND
echo_var REPORT_ADDR
echo_var WORKER_PORT
echo_var WORKSPACE_DIR
echo_var SERVER_DIR
echo_var ENV_PATH
echo_var DEBUG_LOG
echo_var PYWORKER_LOG
echo_var MODEL_LOG

env | grep _ >> /etc/environment;


if [ ! -d "$ENV_PATH" ]
then
    apt install -y python3.10-venv g++ ccache vim
    echo "setting up venv"
    git clone https://github.com/JadarTheObscurity/pyworker "$SERVER_DIR"
    cd $SERVER_DIR
    git checkout -b llama.cpp
    git pull origin llama.cpp

    python3 -m venv "$WORKSPACE_DIR/worker-env"
    source "$WORKSPACE_DIR/worker-env/bin/activate"

    pip install -r requirements.txt

    # Download model from Hugging Face
    huggingface-cli login --token $HF_TOKEN
    mkdir -p /workspace/models && huggingface-cli download $MODEL_ID model.gguf --revision $MODEL_VERSION --local-dir /workspace/models
    mkdir -p /models && ln -s /workspace/models/model.gguf /models/otaku.gguf

    # Download llama.cpp and compile
    /usr/sbin/update-ccache-symlinks
    echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc 
    source ~/.bashrc && echo $PATH
    cd /workspace && git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp && make GGML_CUDA=1 -j `nproc`
    ln -s /workspace/llama.cpp/llama-server /usr/bin/llama-server

    source "$WORKSPACE_DIR/worker-env/bin/activate"
    touch ~/.no_auto_tmux
else
    source "$WORKSPACE_DIR/worker-env/bin/activate"
    echo "environment activated"
    echo "venv: $VIRTUAL_ENV"
fi

[ ! -d "$SERVER_DIR/workers/$BACKEND" ] && echo "$BACKEND not supported!" && exit 1

if [ "$USE_SSL" = true ]; then

    cat << EOF > /etc/openssl-san.cnf
    [req]
    default_bits       = 2048
    distinguished_name = req_distinguished_name
    req_extensions     = v3_req

    [req_distinguished_name]
    countryName         = US
    stateOrProvinceName = CA
    organizationName    = Vast.ai Inc.
    commonName          = vast.ai

    [v3_req]
    basicConstraints = CA:FALSE
    keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName   = @alt_names

    [alt_names]
    IP.1   = 0.0.0.0
EOF

openssl req -newkey rsa:2048 -subj "/C=US/ST=CA/CN=pyworker.vast.ai/" \
    -nodes \
    -sha256 \
    -keyout /etc/instance.key \
    -out /etc/instance.csr \
    -config /etc/openssl-san.cnf

curl --header 'Content-Type: application/octet-stream' \
    --data-binary @//etc/instance.csr \
    -X \
    POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=$CONTAINER_ID" > /etc/instance.crt;
fi




export REPORT_ADDR WORKER_PORT USE_SSL

cd "$SERVER_DIR"

echo "launching PyWorker server"

# if instance is rebooted, we want to clear out the log file so pyworker doesn't read lines
# from the run prior to reboot. past logs are saved in $MODEL_LOG.old for debugging only
[ -e "$MODEL_LOG" ] && cat "$MODEL_LOG" >> "$MODEL_LOG.old" && : > "$MODEL_LOG"

(python3 -m "workers.$BACKEND.server" |& tee -a "$PYWORKER_LOG") &
echo "launching PyWorker server done"
