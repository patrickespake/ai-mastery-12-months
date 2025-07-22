# Complete Guide: AI Development Environment on Manjaro Linux

## 1. Update System

```bash
sudo pacman -Syu
```

## 2. Install Base Dependencies

```bash
# Install essential tools
sudo pacman -S base-devel curl wget git tree htop unzip vim nano
```

**Purpose**: Essential system tools for development, file management, and system monitoring.

## 3. Install Base Python (Optional - Anaconda already includes Python)

```bash
# Install system Python (backup)
sudo pacman -S python python-pip

# Check version
python --version
```

**Purpose**: System-level Python installation as a fallback, though Anaconda will provide the main Python environment.

## 4. Install Anaconda

```bash
# Navigate to Downloads
cd ~/Downloads

# Download Anaconda (latest version)
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh

# Give execution permission
chmod +x Anaconda3-2025.06-0-Linux-x86_64.sh

# Run installer
bash Anaconda3-2025.06-0-Linux-x86_64.sh

# IMPORTANT: During installation:
# - Accept license (type "yes")
# - Confirm installation location (press Enter for ~/anaconda3)
# - ANSWER "yes" when asked about initializing Anaconda3
```

**Purpose**: Anaconda is a Python distribution that includes package management (conda), virtual environments, and pre-installed scientific libraries. It simplifies dependency management for data science and AI projects.

## 5. Configure Conda in Shell (CRUCIAL STEP)

```bash
# Initialize conda for zsh
~/anaconda3/bin/conda init zsh

# Reload shell
source ~/.zshrc

# OR restart terminal completely
exec zsh

# Verify it worked (should show (base) in prompt)
conda --version
conda info
```

**Purpose**: Configures the shell to recognize conda commands and automatically activate environments.

## 6. Create Python 3.11 Environment for AI

```bash
# IMPORTANT: Use Python 3.11 for maximum compatibility
conda create -n ai-studies python=3.11 -y

# Activate environment
conda activate ai-studies

# Verify it's active (should show (ai-studies) in prompt)
conda info --envs
python --version  # Should show Python 3.11.x
```

**Purpose**: Creates an isolated Python environment specifically for AI work, preventing conflicts between different projects and ensuring reproducible setups.

## 7. Install Jupyter

```bash
# With ai-studies environment activated
conda install -y jupyter jupyterlab notebook ipywidgets

# Useful Jupyter extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Configure Jupyter (optional)
jupyter notebook --generate-config
```

**Purpose**:
- **Jupyter Notebook**: Interactive computing environment for data exploration and prototyping
- **JupyterLab**: Modern web-based interface for Jupyter with enhanced features
- **ipywidgets**: Interactive widgets for building GUIs in notebooks

## 8. Install Base Scientific Libraries

```bash
# Fundamental scientific libraries
conda install -y numpy pandas matplotlib seaborn scikit-learn scipy

# Advanced visualization
conda install -y plotly bokeh

# Statistics and analysis
conda install -y statsmodels
```

**Purpose**:
- **NumPy**: Fundamental package for numerical computing with arrays
- **Pandas**: Data manipulation and analysis library
- **Matplotlib/Seaborn**: Data visualization libraries
- **Scikit-learn**: Machine learning library with algorithms and tools
- **SciPy**: Scientific computing library with advanced mathematical functions
- **Plotly/Bokeh**: Interactive visualization libraries
- **Statsmodels**: Statistical modeling and econometrics

## 9. Install PyTorch (CPU) - Reliable Method

```bash
# Install PyTorch for CPU via pip (more stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print('✓ PyTorch installed for CPU')
"
```

**Purpose**: PyTorch is a deep learning framework that provides dynamic neural networks, automatic differentiation, and GPU acceleration. Excellent for research and production.

## 10. Install TensorFlow

```bash
# Install TensorFlow (CPU)
pip install tensorflow

# Verify installation
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print('✓ TensorFlow installed')
"
```

**Purpose**: TensorFlow is Google's deep learning framework, excellent for production deployment, mobile applications, and large-scale distributed training.

## 11. Install Essential AI and ML Libraries

```bash
# Computer Vision
pip install opencv-python pillow

# Natural Language Processing
pip install nltk spacy

# Hugging Face Transformers
pip install transformers datasets huggingface-hub tokenizers

# MLOps and Experiments
pip install wandb mlflow optuna

# Gradient Boosting
pip install xgboost lightgbm catboost

# Other useful libraries
pip install \
    networkx \
    beautifulsoup4 \
    requests \
    streamlit \
    gradio \
    fastapi \
    uvicorn \
    joblib \
    tqdm
```

**Purpose**:
- **OpenCV/Pillow**: Computer vision and image processing
- **NLTK/spaCy**: Natural language processing and text analysis
- **Transformers**: State-of-the-art pre-trained models (BERT, GPT, etc.)
- **Weights & Biases/MLflow**: Experiment tracking and model management
- **Optuna**: Hyperparameter optimization
- **XGBoost/LightGBM/CatBoost**: Gradient boosting frameworks
- **NetworkX**: Graph analysis and network science
- **Streamlit/Gradio**: Quick web app development for ML demos
- **FastAPI**: Modern web framework for building APIs

## 12. Install and Configure Git

```bash
# Check if Git is installed
git --version

# Configure global identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Useful configurations
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"

# Generate SSH key for GitHub (optional)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start ssh-agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Show public key to add to GitHub
echo "Copy this key to GitHub:"
cat ~/.ssh/id_ed25519.pub
```

**Purpose**: Git is essential for version control, collaboration, and code management. SSH keys enable secure authentication with remote repositories.

## 13. Install VS Code

```bash
# Option 1: Via AUR (recommended for Manjaro)
yay -S visual-studio-code-bin

# Verify installation
code --version
```

**Purpose**: VS Code is a powerful, extensible code editor with excellent Python support, integrated terminal, debugging capabilities, and extensive plugin ecosystem.

## 14. Configure VS Code for Python/AI

```bash
# Install essential extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.flake8
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension ms-toolsai.vscode-jupyter-slideshow
code --install-extension gitpod.gitpod-desktop
code --install-extension github.copilot
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode.live-server
code --install-extension eamodio.gitlens
```

**Purpose**:
- **Python extension**: Core Python support with IntelliSense, debugging, and linting
- **Black/isort**: Code formatting and import organization
- **Jupyter extensions**: Notebook support within VS Code
- **GitHub Copilot**: AI-powered code completion
- **GitLens**: Enhanced Git integration and history visualization

## 15. Configure VS Code Settings

```bash
# Create configuration directory
mkdir -p ~/.config/Code/User

# Create configuration file
cat > ~/.config/Code/User/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/ai-studies/bin/python",
    "python.terminal.activateEnvironment": true,
    "jupyter.askForKernelRestart": false,
    "jupyter.alwaysTrustNotebooks": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.black-formatter"
    },
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "files.autoSave": "afterDelay",
    "workbench.colorTheme": "Default Dark+",
    "terminal.integrated.defaultProfile.linux": "zsh",
    "editor.fontSize": 14,
    "editor.tabSize": 4,
    "editor.insertSpaces": true
}
EOF
```

**Purpose**: Configures VS Code for optimal Python/AI development with automatic formatting, linting, and integration with conda environments.

## 16. Additional Useful Tools

```bash
# Docker for containerization
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# PostgreSQL
sudo pacman -S postgresql
sudo systemctl enable --now postgresql

# Redis
sudo pacman -S redis
sudo systemctl enable --now redis

# Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable --py widgetsnbextensions
```

**Purpose**:
- **Docker**: Containerization for reproducible deployments and isolated environments
- **PostgreSQL**: Robust relational database for data storage
- **Redis**: In-memory data store for caching and real-time applications
- **Jupyter extensions**: Additional notebook functionality and widgets

## 17. Create Project Structure

```bash
# Create main directory
mkdir -p ~/Projects/AI-Studies
cd ~/Projects/AI-Studies

# Create folder structure
mkdir -p {notebooks,datasets,models,scripts,docs,experiments,resources}

# Create README
cat > README.md << 'EOF'
# AI Studies Environment

Environment configured for Artificial Intelligence and Machine Learning studies.

## Project Structure
- `notebooks/`: Jupyter notebooks for experiments
- `datasets/`: Data sets
- `models/`: Trained models
- `scripts/`: Python scripts
- `docs/`: Documentation
- `experiments/`: Experiments and tests
- `resources/`: Additional resources

## Environment Activation
conda activate ai-studies
EOF
```

# Create .gitignore


```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Data
datasets/
*.csv
*.json
*.pkl
*.h5

# Models
models/*.pkl
models/*.h5
models/*.pt

# IDEs
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF
```

**Purpose**: Creates an organized project structure following best practices for AI/ML projects with proper version control setup.

## 18. Create Test Notebook

```bash
# Navigate to notebooks folder
cd ~/Projects/AI-Studies/notebooks
```

```bash
# Create installation test notebook
cat > test_environment.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AI Environment Test\n",
    "\n",
    "Verification of main installed libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import main libraries\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import sklearn\n",
    "import torch\n",
    "import tensorflow as tf\n",
    "\n",
    "print(\"=== Environment Verification ===\")\n",
    "print(f\"Python: {sys.version}\")\n",
    "print(f\"NumPy: {np.__version__}\")\n",
    "print(f\"Pandas: {pd.__version__}\")\n",
    "print(f\"Matplotlib: {plt.matplotlib.__version__}\")\n",
    "print(f\"Seaborn: {sns.__version__}\")\n",
    "print(f\"Scikit-learn: {sklearn.__version__}\")\n",
    "print(f\"PyTorch: {torch.__version__}\")\n",
    "print(f\"TensorFlow: {tf.__version__}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualization test\n",
    "plt.figure(figsize=(12, 4))\n",
    "\n",
    "# Subplot 1: NumPy\n",
    "plt.subplot(1, 3, 1)\n",
    "data = np.random.randn(1000)\n",
    "plt.hist(data, bins=30, alpha=0.7, color='blue')\n",
    "plt.title('Normal Distribution (NumPy)')\n",
    "plt.xlabel('Value')\n",
    "plt.ylabel('Frequency')\n",
    "\n",
    "# Subplot 2: Pandas\n",
    "plt.subplot(1, 3, 2)\n",
    "df = pd.DataFrame({\n",
    "    'x': np.random.randn(100),\n",
    "    'y': np.random.randn(100)\n",
    "})\n",
    "plt.scatter(df['x'], df['y'], alpha=0.6, color='red')\n",
    "plt.title('Scatter Plot (Pandas)')\n",
    "plt.xlabel('X')\n",
    "plt.ylabel('Y')\n",
    "\n",
    "# Subplot 3: Seaborn\n",
    "plt.subplot(1, 3, 3)\n",
    "tips = sns.load_dataset('tips')\n",
    "sns.boxplot(data=tips, x='day', y='total_bill')\n",
    "plt.title('Box Plot (Seaborn)')\n",
    "plt.xticks(rotation=45)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PyTorch test\n",
    "print(\"=== PyTorch Test ===\")\n",
    "x = torch.randn(3, 3)\n",
    "y = torch.randn(3, 3)\n",
    "z = torch.matmul(x, y)\n",
    "print(f\"Tensor X: {x.shape}\")\n",
    "print(f\"Tensor Y: {y.shape}\")\n",
    "print(f\"Matrix multiplication X@Y: {z.shape}\")\n",
    "print(\"✓ PyTorch working\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TensorFlow test\n",
    "print(\"=== TensorFlow Test ===\")\n",
    "a = tf.constant([[1.0, 2.0], [3.0, 4.0]])\n",
    "b = tf.constant([[1.0, 1.0], [0.0, 1.0]])\n",
    "c = tf.matmul(a, b)\n",
    "print(f\"Tensor A: {a.shape}\")\n",
    "print(f\"Tensor B: {b.shape}\")\n",
    "print(f\"Matrix multiplication A@B: {c.shape}\")\n",
    "print(\"✓ TensorFlow working\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scikit-learn test\n",
    "from sklearn.datasets import make_classification\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "print(\"=== Scikit-learn Test ===\")\n",
    "\n",
    "# Create synthetic dataset\n",
    "X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train model\n",
    "clf = RandomForestClassifier(random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = clf.predict(X_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "print(f\"Dataset: {X.shape}\")\n",
    "print(f\"Accuracy: {accuracy:.3f}\")\n",
    "print(\"✓ Scikit-learn working\")\n",
    "\n",
    "print(\"\\n🎉 AI environment configured successfully!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
```

**Purpose**: Provides a comprehensive test notebook to verify all installed libraries are working correctly and demonstrate basic functionality.

## 19. Automated Verification Script

```bash
# Create complete verification script
cat > ~/check_ai_environment.sh << 'EOF'
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

EOF

chmod +x ~/check_ai_environment.sh
```

**Purpose**: Automated script to verify all components are installed and working correctly.

## 20. Useful Commands and Aliases

```bash
# Add useful aliases to .zshrc
cat >> ~/.zshrc << 'EOF'

# === AI Development Aliases ===
alias ai="conda activate ai-studies"
alias ai-jupyter="conda activate ai-studies && cd ~/Projects/AI-Studies && jupyter lab"
alias ai-code="conda activate ai-studies && cd ~/Projects/AI-Studies && code ."
alias ai-check="~/check_ai_environment.sh"
alias ai-update="conda activate ai-studies && pip list --outdated"
alias ai-notebook="conda activate ai-studies && cd ~/Projects/AI-Studies/notebooks && jupyter notebook"

# General shortcuts
alias ll="ls -la"
alias la="ls -A"
alias l="ls -CF"
alias ..="cd .."
alias ...="cd ../.."

EOF

# Reload configuration
source ~/.zshrc
```

**Purpose**: Creates convenient shortcuts for common AI development tasks and navigation.

## 21. Final Installation Test

```bash
# Run verification
~/check_ai_environment.sh

# Test alias commands
ai  # Should activate environment

# Start Jupyter Lab
ai-jupyter

# In another terminal, open VS Code
ai-code
```

**Purpose**: Validates that the entire setup is working correctly.

## 22. Daily Usage Commands

### Environment Management
```bash
# Activate environment
ai  # or conda activate ai-studies

# List installed packages
conda list

# Install new library
pip install library-name

# Update libraries
pip install --upgrade library-name

# Deactivate environment
conda deactivate
```

### Development
```bash
# Start Jupyter Lab
ai-jupyter

# Start VS Code in project
ai-code

# Check environment
ai-check

# Create new notebook
cd ~/Projects/AI-Studies/notebooks
jupyter notebook new_notebook.ipynb
```

### Git
```bash
# Initialize repository
cd ~/Projects/AI-Studies
git init
git add .
git commit -m "Initial setup"

# Connect with GitHub
git remote add origin https://github.com/your-username/ai-studies.git
git push -u origin main
```

## ✅ Summary

You now have a complete AI development environment configured on Manjaro Linux with:

- ✅ **Python 3.11** (maximum compatibility)
- ✅ **Anaconda** for environment management
- ✅ **Jupyter Lab** for interactive notebooks
- ✅ **PyTorch** (CPU) working correctly
- ✅ **TensorFlow** for deep learning
- ✅ **Scikit-learn** for classical machine learning
- ✅ **VS Code** configured for Python/AI
- ✅ **Git** configured for version control
- ✅ **Organized project structure**
- ✅ **Verification scripts and useful aliases**

## Tool Overview Summary

| Tool | Purpose |
|------|---------|
| **Anaconda** | Python distribution with package and environment management |
| **Jupyter Lab** | Interactive computing environment for data science |
| **PyTorch** | Dynamic deep learning framework for research and production |
| **TensorFlow** | Production-ready deep learning framework with deployment focus |
| **Scikit-learn** | Classical machine learning algorithms and tools |
| **NumPy/Pandas** | Numerical computing and data manipulation |
| **Matplotlib/Seaborn** | Data visualization and statistical plotting |
| **VS Code** | Modern code editor with extensive AI/ML extensions |
| **Git** | Version control for code and collaboration |
| **Docker** | Containerization for reproducible deployments |

To start working, use these commands:
- `ai` - activate environment
- `ai-jupyter` - start Jupyter Lab
- `ai-code` - open VS Code
- `ai-check` - verify installation
