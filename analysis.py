import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def perform_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model

def plot_regression_results(data, model, title, coefficient_label):
    # Calculate frequency of each data point
    frequency = data.groupby(['coeff', 'points']).size().reset_index(name='frequency')
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with point size based on frequency
    sns.scatterplot(x='coeff', y='points', size='frequency', sizes=(20, 200), data=frequency, legend='brief')
    
    # Plot regression line
    plt.plot(data['coeff'], model.predict(sm.add_constant(data[['coeff']])), color='red', label='Regression Line')
    
    # Add confidence interval
    X_pred = sm.add_constant(np.linspace(data['coeff'].min(), data['coeff'].max(), 100))
    y_pred = model.get_prediction(X_pred).summary_frame()
    plt.fill_between(X_pred[:, 1], y_pred['mean_ci_lower'], y_pred['mean_ci_upper'], color='red', alpha=0.2, label='Confidence Interval')
    
    # Add trend lines (polynomial regression of degree 2)
    poly_model = np.poly1d(np.polyfit(data['coeff'], data['points'], 2))
    plt.plot(data['coeff'], poly_model(data['coeff']), color='green', label='Trend Line (Degree 2)')
    
    # Customize aesthetics
    plt.title(title, fontsize=16)
    plt.xlabel(f"{coefficient_label} Coefficient", fontsize=12)
    plt.ylabel('Points', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  
    # Place legend outside the plot area
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlapping
    
    # Include important statistics results near the plot as annotations
    coeff_value = model.params['coeff']
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    pve = calculate_pve(model)

    plt.text(1.01, 1, f"PVE): {pve:.4f}$", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(1.01, 0.95, f"Coefficient Value: {coeff_value:.4f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(1.01, 0.90, f"R-squared: {r_squared:.4f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(1.01, 0.85, f"Adj. R-squared: {adj_r_squared:.4f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    plt.show()


def calculate_pve (model):
    # Calculate the total sum of squares (TSS)
    tss = np.sum((model.model.endog - np.mean(model.model.endog)) ** 2)
    
    # Calculate the residual sum of squares (RSS)
    rss = np.sum(model.resid ** 2)
    
    # Calculate the proportion of variance explained (PVE)
    pve = 1 - (rss / tss)

    return pve

def analyze_coefficients (model):
    pve = calculate_pve(model)

    coeff_value = model.params['coeff']

    return pve, coeff_value

def process_results(data, title,coefficient_label):
    X = data[['coeff']]
    y = data['points']
    model = perform_regression(X, y)

    plot_regression_results(data, model, title, coefficient_label)
    pve, coeff_value = analyze_coefficients(model)

    print(title)
    print(f"Coefficient Value: {coeff_value:.4f}")
    print(f"Percentage Variance Explained (PVE): {pve * 100:.4f}%")
    print()


def read_json_to_df(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return pd.DataFrame(data)

# File paths
border_file = 'border_coeff_points.json'
language_file = 'language_coeff_points.json'
political_file = 'political_coeff_points.json'

# Read JSON files into DataFrames
border_df = read_json_to_df(border_file)
language_df = read_json_to_df(language_file)
political_df = read_json_to_df(political_file)


# Data for Border Coefficient
border_jury_data = border_df[border_df['jury_or_televoting'] == 'J']
border_telovoting_data = border_df[border_df['jury_or_televoting'] == 'T']

print("____Border Coefficient____")
process_results(border_jury_data, 'Jury', "Border")
process_results(border_telovoting_data, 'Telovoting', "Border")


# Data for Language Coefficient
language_jury_data = language_df[language_df['jury_or_televoting'] == 'J']
language_telovoting_data = language_df[language_df['jury_or_televoting'] == 'T']

print("____Language Coefficient____")
process_results(language_jury_data, 'Jury', "Language")
process_results(language_telovoting_data, 'Telovoting', "Language")


# Data for Political Coefficient
political_jury_data = political_df[political_df['jury_or_televoting'] == 'J']
political_telovoting_data = political_df[political_df['jury_or_televoting'] == 'T']

print("____Political Coefficient____")
process_results(political_jury_data, 'Jury', "Political")
process_results(political_telovoting_data, 'Telovoting', "Political")
