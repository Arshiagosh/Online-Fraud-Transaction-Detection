# Online Fraud Transaction Detection

This repository contains a Jupyter Notebook that focuses on developing a decision tree algorithm to detect fraudulent online transactions. Fraudulent activities in virtual payments pose significant risks to financial institutions and security systems, and this project aims to address this challenge using machine learning techniques.

## Description

The notebook covers the end-to-end process of building a decision tree model for fraud detection, including data preprocessing, feature engineering, model training, and performance evaluation. Two popular splitting criteria, entropy and Gini index, are explored for constructing the decision tree.

By exploring this notebook, readers can gain insights into the application of decision trees in fraud detection scenarios and understand the differences in performance between the entropy and Gini index as impurity measures.

## Features

- **Data Preprocessing**: The notebook includes techniques for data cleaning, handling missing values, and removing duplicates.
- **Feature Engineering**: Techniques such as one-hot encoding and binning are employed to transform categorical and continuous features, respectively.
- **Model Training**: The decision tree algorithm is implemented from scratch, and various modeling approaches are explored.
- **Evaluation**: The performance of the trained models is evaluated using confusion matrices, accuracy, precision, recall, and F1-score.
- **Visualization**: The notebook includes visualizations of the correlation matrix, confusion matrix and tree itself for better understanding of the data and model performance.

## Requirements

To run the Jupyter Notebook, you'll need to have the following libraries installed:

- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost
- GraphViz
- pydotplus

You can install these libraries using pip:
```python
pip install -r requirements.txt
```
## Usage

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Open the Jupyter Notebook file (`main.ipynb`) in your preferred environment (e.g., Jupyter Notebook, JupyterLab, or Visual Studio Code with the Python extension).
4. Execute the code cells sequentially to follow the step-by-step implementation and analysis.
5. The produced graphs are in `./Graphs`

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
