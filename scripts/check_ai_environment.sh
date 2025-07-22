#!/bin/bash

echo "🔍 === Complete AI Environment Verification ==="
echo

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai-studies

echo "1️⃣  Operating System:"
uname -a
echo

echo "2️⃣  Python and Conda:"
python --version
conda --version
echo "Active environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo

echo "3️⃣  Main Libraries:"
python -c "
try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
except ImportError:
    print('✗ NumPy not installed')

try:
    import pandas as pd
    print(f'✓ Pandas: {pd.__version__}')
except ImportError:
    print('✗ Pandas not installed')

try:
    import matplotlib
    print(f'✓ Matplotlib: {matplotlib.__version__}')
except ImportError:
    print('✗ Matplotlib not installed')

try:
    import sklearn
    print(f'✓ Scikit-learn: {sklearn.__version__}')
except ImportError:
    print('✗ Scikit-learn not installed')

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'  - CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('✗ PyTorch not installed')

try:
    import tensorflow as tf
    print(f'✓ TensorFlow: {tf.__version__}')
except ImportError:
    print('✗ TensorFlow not installed')
"
echo

echo "4️⃣  Jupyter:"
jupyter --version | head -1
echo

echo "5️⃣  VS Code:"
code --version | head -1
echo

echo "6️⃣  Git:"
git --version
echo

echo "7️⃣  Project Structure:"
if [ -d "$HOME/Projects/AI-Studies" ]; then
    echo "✓ Project directory created"
    ls -la ~/Projects/AI-Studies/
else
    echo "✗ Project directory not found"
fi
echo

echo "🎉 === Verification Complete ==="
echo "To get started:"
echo "1. conda activate ai-studies"
echo "2. cd ~/Projects/AI-Studies"
echo "3. jupyter lab"
echo "4. code . (in another terminal)"

