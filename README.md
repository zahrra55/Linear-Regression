# Linear Regression with Gradient Descent

**Author:** Zahraa Ibrahim  
**Course:** Large Scale Machine Learning (LSML)  
**Institution:** Al-Nahrain University  
**Program:** Master of Science in Computer Science (MSCS) - First Year  
**Assignment:** Step-by-Step Linear Regression Implementation

## Project Overview

This project demonstrates the implementation of Linear Regression using Gradient Descent from scratch, with detailed step-by-step visualization of how the algorithm learns and converges to the optimal solution. The implementation was developed as part of the Large Scale Machine Learning course requirements.

## Learning Objectives

- **Understanding Linear Regression**: Implement linear regression from first principles
- **Gradient Descent Algorithm**: Step-by-step implementation of the gradient descent optimization
- **Visual Learning Process**: Demonstrate how the regression line evolves during training
- **Mathematical Foundation**: Show the mathematical calculations behind each step
- **Real-world Application**: Apply the algorithm to the California Housing dataset

## Dataset

**California Housing Dataset**
- **Features**: Median Income (MedInc) in $10,000s
- **Target**: Median House Value (MedHouseVal) in $100,000s
- **Source**: Scikit-learn's built-in dataset
- **Preprocessing**: Added bias term for intercept calculation

## Implementation Details

### Mathematical Foundation

The linear regression model is defined as:
```
h_θ(x) = θ₀ + θ₁x
```

Where:
- `θ₀` (theta_0): Intercept term (bias)
- `θ₁` (theta_1): Slope coefficient
- `x`: Input feature (Median Income)

### Cost Function (Mean Squared Error)
```
J(θ) = (1/2m) * Σ(h_θ(xⁱ) - yⁱ)²
```

### Gradient Descent Update Rule
```
θⱼ := θⱼ - α * (∂J/∂θⱼ)
```

Where:
- `α` (alpha): Learning rate = 0.01
- `m`: Number of training examples

### Partial Derivatives
```
∂J/∂θ₀ = (1/m) * Σ(h_θ(xⁱ) - yⁱ)
∂J/∂θ₁ = (1/m) * Σ(h_θ(xⁱ) - yⁱ) * xⁱ
```

## Key Features

### 1. **Step-by-Step Visualization**
- **5 Progressive Stages**: Shows regression line evolution
- **Iteration Checkpoints**: 1, 50, 200, 1000, and 2000 iterations
- **Real-time Equation Display**: Current model parameters shown on each plot
- **Convergence Tracking**: Visual demonstration of how the algorithm learns

### 2. **Mathematical Transparency**
- **Vectorized Implementation**: Efficient matrix operations
- **Gradient Calculations**: Explicit computation of partial derivatives
- **Parameter Updates**: Clear demonstration of weight updates
- **Cost Function**: Implicit optimization of Mean Squared Error

### 3. **Professional Visualization**
- **Clean Plot Styling**: Seaborn-based professional appearance
- **Clear Labels**: Descriptive axis labels and titles
- **Color Coding**: Distinct colors for data points and regression lines
- **Annotation**: Current model equation displayed on each plot

## Results

### Final Model Parameters
- **Intercept (θ₀)**: 0.3898
- **Slope (θ₁)**: 0.4308
- **Final Equation**: `y = 0.4308x + 0.3898`

### Training Configuration
- **Learning Rate**: 0.01
- **Total Iterations**: 2000
- **Convergence**: Achieved stable parameters
- **Training Data**: 80% of California Housing dataset

## Technical Implementation

### Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import matplotlib as mpl
```

### Key Components
1. **Data Loading & Preprocessing**
   - California Housing dataset
   - Feature selection (Median Income)
   - Bias term addition
   - Train/test split (80/20)

2. **Gradient Descent Algorithm**
   - Random parameter initialization
   - Iterative parameter updates
   - Convergence monitoring
   - Step-by-step visualization

3. **Visualization System**
   - Progressive learning visualization
   - Real-time parameter display
   - Professional plot styling
   - Clear progression tracking

## Educational Value

This implementation serves as an excellent educational tool for understanding:

- **Machine Learning Fundamentals**: Core concepts of supervised learning
- **Optimization Theory**: How gradient descent minimizes cost functions
- **Mathematical Implementation**: From theory to practical code
- **Visual Learning**: Seeing algorithms work in real-time
- **Professional Development**: Clean, well-documented code practices

## Academic Context

**Course**: Large Scale Machine Learning (LSML)  
**Supervisor**: Dr. B. N. Dhannoon  
**University**: Al-Nahrain University  
**Student**: Zahraa Ibrahim  
**Program**: Master of Science in Computer Science (MSCS)  
**Year**: First Year Graduate Student  

## Project Structure

```
Linear-Regression/
├── README.md                 # This documentation
├── LSML_LR_GD.ipynb         # Main implementation notebook
└── LICENSE                  # Project license
```

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required packages: numpy, matplotlib, scikit-learn

### Running the Code
1. Clone the repository
2. Open `LSML_LR_GD.ipynb` in Jupyter Notebook
3. Run all cells to see the step-by-step gradient descent process
4. Observe how the regression line evolves through different stages

### Alternative: Google Colab
Click the "Open in Colab" badge at the top of the notebook for cloud-based execution.

## Key Learning Outcomes

1. **Algorithm Understanding**: Deep comprehension of gradient descent mechanics
2. **Mathematical Implementation**: Practical application of calculus in ML
3. **Visualization Skills**: Creating informative plots for algorithm analysis
4. **Code Organization**: Professional development practices
5. **Academic Writing**: Clear documentation and explanation

## Notes

This project was developed in response to the professor's request to demonstrate step-by-step calculation of linear regression outputs. The implementation shows not just the final result, but the entire learning process, making it an excellent educational resource for understanding how machine learning algorithms actually work.

---

**Contact**: Zahraa Ibrahim  
**Institution**: Al-Nahrain University  
**Course**: Large Scale Machine Learning  
**Supervisor**: Dr. B. N. Dhannoon