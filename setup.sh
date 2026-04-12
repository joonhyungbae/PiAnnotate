#!/bin/bash

# PiAnnotate - Setup Pipeline
# Sets up the conda environment and installs dependencies.
# FurElise dataset and MANO models must be obtained separately
# (see README.md for instructions).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$SCRIPT_DIR/setup"

echo "PiAnnotate - Setup Pipeline"
echo "   Project path: $SCRIPT_DIR"
echo ""

if [ ! -d "$SETUP_DIR" ]; then
    echo "Setup directory not found: $SETUP_DIR"
    exit 1
fi

cd "$SCRIPT_DIR"

# ============================================
# Environment Setup (conda + npm)
# ============================================
echo "=========================================="
echo "Setting up environment..."
echo "=========================================="
echo ""

ENV_SCRIPT="$SETUP_DIR/env.sh"
if [ -f "$ENV_SCRIPT" ]; then
    bash "$ENV_SCRIPT"
else
    echo "Environment setup script not found: $ENV_SCRIPT"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Before running, make sure you have:"
echo "  1. Downloaded FurElise dataset to for_elise/"
echo "     (see README.md for instructions)"
echo "  2. Downloaded MANO v1.2 model to web/mano_v1_2/"
echo "     (from https://mano.is.tue.mpg.de/)"
echo ""
echo "To run: bash start.sh"
echo "=========================================="
