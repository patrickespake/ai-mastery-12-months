# Scripts Directory

This directory contains automation scripts and data files for the AI Mastery 12-Month learning journey project.

## Scripts

### check_ai_environment.sh
A comprehensive bash script that verifies the complete AI development environment setup. The script performs the following checks:

- **Operating System**: Displays system information using `uname`
- **Python and Conda**: Verifies Python version, Conda version, and shows the active environment
- **Main Libraries**: Checks the installation and versions of essential AI/ML libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - PyTorch (including CUDA availability)
  - TensorFlow
- **Jupyter**: Verifies Jupyter installation
- **VS Code**: Checks Visual Studio Code installation
- **Git**: Verifies Git version
- **Project Structure**: Checks if the AI-Studies project directory exists

The script automatically activates the `ai-studies` conda environment and provides a summary of the verification results with instructions on how to get started.

### github_importer_english.py
A Python script that automates the import of milestones and issues to GitHub Projects. This tool is designed to set up a complete GitHub project structure for the AI learning journey. Features include:

- **Milestone Creation**: Imports milestones with titles, descriptions, and due dates
- **Issue Creation**: Imports issues with titles, bodies, milestone associations, and labels
- **Label Setup**: Creates a comprehensive label system including:
  - Activity type labels (theory, practice, project, networking)
  - Knowledge area labels (mathematics, programming, machine-learning, deep-learning, ai-generative)
  - Priority labels (high, medium, low)
  - Setup and environment labels
  - Weekly labels (week-1 through week-48)
- **Rate Limiting**: Implements proper rate limiting to respect GitHub API limits
- **Interactive Menu**: Provides options to:
  1. Setup labels
  2. Import milestones and issues from CSV files
  3. Create milestones manually
  4. Exit

The script requires GitHub credentials (username, repository name, and personal access token) and uses the GitHub API v3.

## Data Files

### all_issues_complete_english.csv
A comprehensive CSV file containing all issues for the AI learning journey. Each row represents a task with the following fields:

- **title**: The issue title
- **body**: Detailed description of the task
- **milestone**: Associated milestone (e.g., "MILESTONE 1: Fundamentals and Mathematics")
- **labels**: Comma-separated list of labels
- **assignee**: GitHub username (typically @you)
- **estimate**: Time estimate in hours
- **priority**: Task priority (High, Medium, Low)
- **due_date**: Expected completion date
- **state**: Issue state (open/closed)

The file contains hundreds of carefully structured tasks covering topics from basic setup through advanced AI projects, organized across 48 weeks of study.

### milestones_import_english.csv
A CSV file containing the six major milestones for the AI learning journey:

1. **MILESTONE 1: Fundamentals and Mathematics** (Due: 2025-08-15)
   - Foundation in mathematics, Python programming, and fundamental AI concepts

2. **MILESTONE 2: Classical Machine Learning** (Due: 2025-10-15)
   - Traditional ML algorithms and validation techniques

3. **MILESTONE 3: Deep Learning and Neural Networks** (Due: 2025-12-15)
   - Neural networks, CNNs, RNNs, and advanced architectures

4. **MILESTONE 4: Generative AI and Advanced Models** (Due: 2026-02-15)
   - GANs, VAEs, Diffusion Models, and Large Language Models

5. **MILESTONE 5: Specialization and Applications** (Due: 2026-04-15)
   - Domain-specific applications and production deployment

6. **MILESTONE 6: Advanced Projects and Portfolio** (Due: 2026-06-15)
   - Professional portfolio and complex capstone projects

## Usage

1. **Environment Verification**: Run `./check_ai_environment.sh` to verify your development environment is properly set up.

2. **GitHub Project Setup**: Use `python github_importer_english.py` to automatically create your GitHub project structure with all milestones, issues, and labels.

Both scripts are designed to work together to provide a smooth setup experience for the AI Mastery learning journey.