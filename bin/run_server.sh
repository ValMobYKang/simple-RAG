#!/bin/bash

set -e

# Initial 
GPU=${1:-1} # 0 - cpu / 1 - gpu m1
source .env
function check_answer() {
    read -p "$1 (yes/no): " answer;
    if [[ "$answer" != "yes" && "$answer" != "y" ]]; then
        echo "Exiting."
        exit;
    fi
}

# Load or download file 
if [ ! -f $MODEL ]; then
    echo "Model not found."
    check_answer "Do you want to download the model?"
    MODEL="./dolphin-2.1-mistral-7b.Q5_K_M.gguf"
    wget "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/blob/main/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"
fi

# Check the model file type
filename=$(basename -- "$MODEL")
if [[ "${filename##*.}" != "gguf" ]]; then 
    echo "Model suffix is not 'gguf'. Exiting."
    exit;
fi

# Check the .venv folder
if [ ! -d '.venv' ]; then
    echo "'venv' not found."
    check_answer "Do you want to create '.venv' environment?"
    python3 -m venv .venv
fi

# Activate the venv and check the env
source .venv/bin/activate
if [[  $(which python3) != *"$(pwd)"* ]]; then
    echo "Error: current $(which python3) and target $(pwd) does not match."
    echo "Exiting."
    exit;
fi

# Install llama-cpp-python lib
if ! pip list 2>/dev/null | grep -q 'llama_cpp_python'; then
    echo "llama_cpp_python not is installed."
    check_answer "Do you want to install llama_cpp_python?"
    if [ $GPU -eq 0 ]; then
        pip install llama-cpp-python
    else
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
        pip install 'llama-cpp-python[server]'
    fi
fi

# Execute
echo "Start server now ..."
python3 -m llama_cpp.server --model $MODEL --n_gpu_layers $GPU --n_ctx 20000 --verbose False

