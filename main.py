---
canonical_link: "https://medium.com/p/915736acfa29"
date_exported_from_medium: "November 10, 2025"
---

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Load the dataset
    data = pd.read_csv('public_health_data.csv')

    # Display the first few rows
    data.head()

    # Summary statistics
    data.describe()

    # Visualize the distribution of hospital visits
    sns.histplot(data['hospital_visits'], bins=20, kde=False, color='blue')
    plt.title('Distribution of Hospital Visits')
    plt.xlabel('Number of Visits')
    plt.ylabel('Frequency')
    plt.savefig('hospital_visits_distribution.png')
    plt.show()

    # Check for correlations
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.show()

    # Define the formula for Poisson regression
    formula = 'hospital_visits ~ intervention + age + income + urban'

    # Fit the Poisson regression model
    poisson_model = smf.poisson(formula=formula, data=data).fit()

    # Display the summary of the model
    print(poisson_model.summary())

    # Predicted values and residuals
    data['predicted'] = poisson_model.predict()
    data['residuals'] = data['hospital_visits'] - data['predicted']

    # Plot residuals
    sns.scatterplot(x=data['predicted'], y=data['residuals'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig('residuals_vs_predicted.png')
    plt.show()

    # Goodness of fit using deviance
    print(f'Deviance: {poisson_model.deviance}')
    print(f'Degrees of Freedom: {poisson_model.df_resid}')


if __name__ == '__main__':
    main()
