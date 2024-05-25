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
    frequency = data.groupby(['coeff', 'points']).size().reset_index(name='frequency')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='coeff', y='points', size='frequency', sizes=(20, 200), data=frequency, legend='brief')
    plt.plot(data['coeff'], model.predict(sm.add_constant(data[['coeff']])), color='red', label='Regression Line')
    X_pred = sm.add_constant(np.linspace(data['coeff'].min(), data['coeff'].max(), 100))
    y_pred = model.get_prediction(X_pred).summary_frame()
    plt.fill_between(X_pred[:, 1], y_pred['mean_ci_lower'], y_pred['mean_ci_upper'], color='red', alpha=0.2, label='Confidence Interval')
    poly_model = np.poly1d(np.polyfit(data['coeff'], data['points'], 2))
    plt.plot(data['coeff'], poly_model(data['coeff']), color='green', label='Trend Line (Degree 2)')
    plt.title(title, fontsize=16)
    plt.xlabel(f"{coefficient_label} Coefficient", fontsize=12)
    plt.ylabel('Points', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    coeff_value = model.params['coeff']
    plt.text(1.01, 0.95, f"Coefficient Value: {coeff_value:.4f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
  

def model_diagnostics(model):
    residuals = model.resid

    sm.qqplot(residuals, line='45')
    plt.title('Q-Q Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_pve (model):
    tss = np.sum((model.model.endog - np.mean(model.model.endog)) ** 2)
    rss = np.sum(model.resid ** 2)
    pve = 1 - (rss / tss)
    return pve

def analyze_coefficients (model):
    pve = calculate_pve(model)
    coeff_value = model.params['coeff']
    return pve, coeff_value

def process_results(data, title, coefficient_label):
    X = data[['coeff']]
    y = data['points']
    model = perform_regression(X, y)
    plot_regression_results(data, model, title, coefficient_label)
   
    coeff_value = analyze_coefficients(model)
    print(f"{title} (OLS)")
    print(f"Coefficient Value: {coeff_value:.4f}")
  

def read_json_to_df(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

border_file = 'border_coeff_points.json'
language_file = 'language_coeff_points.json'
political_file = 'political_coeff_points.json'
border_df = read_json_to_df(border_file)
language_df = read_json_to_df(language_file)
political_df = read_json_to_df(political_file)
border_jury_data = border_df[border_df['jury_or_televoting'] == 'J']
border_telovoting_data = border_df[border_df['jury_or_televoting'] == 'T']
language_jury_data = language_df[language_df['jury_or_televoting'] == 'J']
language_telovoting_data = language_df[language_df['jury_or_televoting'] == 'T']
political_jury_data = political_df[political_df['jury_or_televoting'] == 'J']
political_telovoting_data = political_df[political_df['jury_or_televoting'] == 'T']

print("____Border Coefficient____")
process_results(border_jury_data, 'Jury', "Border")
process_results(border_telovoting_data, 'Telovoting', "Border")
print("____Language Coefficient____")
process_results(language_jury_data, 'Jury', "Language")
process_results(language_telovoting_data, 'Telovoting', "Language")
print("____Political Coefficient____")
process_results(political_jury_data, 'Jury', "Political")
process_results(political_telovoting_data, 'Telovoting', "Political")
