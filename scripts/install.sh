#!/bin/bash
# Install evaluation-ai from GitHub
set -e

pip install "git+https://github.com/FlossWare/evaluation-ai.git"

echo "evaluation-ai installed successfully"
echo "Verify: python3 -c 'import evaluation_ai; print(evaluation_ai.__version__)'"
