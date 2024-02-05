#!/bin/bash
set -e

# Initial Env & arugments
source .env
source .venv/bin/activate

# Check env
if [[  $(which python3) != *"$(pwd)"* ]]; then
    echo "Error: current $(which python3) and target $(pwd) does not match."
    echo "Exiting."
    exit;
fi

# Check local llm execution
curl -s --head http://localhost:8000 > /dev/null || { echo "Error: execute './start_server' first to initial the LLM server" ; exit 1;}

# Install libs
if python3 -c "import llama-index" &> /dev/null; then
    echo "llama-index is not installed"
    pip install -r requirements.txt
fi

# Execute scripts
python3 src/backend.py

# Read arugments
# while [[ "$#" -gt 0 ]]; do
#     case "$1" in
#         --dev) dev_mode=1; shift;;
#         *) shift;;
#     esac
# done