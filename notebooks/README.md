# Notebooks Directory

This directory contains Jupyter notebooks for the AI Mastery 12-Month learning journey project.

## Notebooks

### test_environment.ipynb

A comprehensive Jupyter notebook designed to verify that the AI development environment is properly configured. The notebook performs the following verifications:

#### 1. Library Import and Version Check
- Imports and displays versions of all essential AI/ML libraries:
  - **Python**: System version information
  - **NumPy**: Numerical computing library
  - **Pandas**: Data manipulation and analysis
  - **Matplotlib**: Plotting library
  - **Seaborn**: Statistical data visualization
  - **Scikit-learn**: Machine learning library
  - **PyTorch**: Deep learning framework
  - **TensorFlow**: Deep learning framework

#### 2. Visualization Test
Creates three subplots to verify visualization capabilities:
- **NumPy Test**: Generates and plots a histogram of normally distributed random data
- **Pandas Test**: Creates a scatter plot from a DataFrame with random data
- **Seaborn Test**: Displays a box plot using the built-in 'tips' dataset

#### 3. PyTorch Test
- Creates random tensors
- Performs matrix multiplication
- Verifies tensor operations are working correctly
- Displays tensor shapes

#### 4. TensorFlow Test
- Creates constant tensors
- Performs matrix multiplication
- Verifies TensorFlow operations
- Displays tensor shapes

#### 5. Scikit-learn Test
- Generates a synthetic classification dataset
- Splits data into training and testing sets
- Trains a Random Forest Classifier
- Evaluates model accuracy
- Confirms the machine learning pipeline is functional

The notebook serves as a quick diagnostic tool to ensure all necessary libraries are installed and functioning correctly before starting the AI learning journey. Upon successful completion, it displays a confirmation message that the AI environment is configured successfully.

## Usage

1. Navigate to the notebooks directory
2. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `test_environment.ipynb`
4. Run all cells to verify your environment setup

If any library fails to import or any test fails, you should revisit the environment setup instructions and ensure all dependencies are properly installed in your conda environment.