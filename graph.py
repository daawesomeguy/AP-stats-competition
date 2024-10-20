import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('four_year_colleges.csv')

# Assuming the target variable is in the last column
# Define the target variable
target = 'default_rate'

# Select features excluding non-numeric columns and OPEID
initial_features = [
    ''
    'median_debt', 'highest_degree', 'ownership', 'locale', 'admit_rate', 'SAT_avg',
    'online_only', 'enrollment', 'net_price', 'avg_cost', 'net_tuition', 'ed_spending_per_student',
    'avg_faculty_salary', 'pct_PELL', 'pct_fed_loan', 'grad_rate', 'pct_firstgen', 
    'med_fam_income', 'med_alum_earnings'
]
'''
features = [col for col in data.columns if col != target and pd.api.types.is_numeric_dtype(data[col]) and col != 'OPEID']



data_filtered = data[features + [target]]  # Select features and target
# Prepare the feature matrix (X) and target vector (y)

X = data_filtered[features]
y = data_filtered[target]
feature_names = X.columns
'''
categorical_features = ['highest_degree', 'ownership', 'locale', 'online_only','region','hbcu']
for feature in categorical_features:
    if feature in data.columns:
        data = pd.get_dummies(data, columns=[feature], drop_first=True)
# Update initial_features after dummification
initial_features = [col for col in data.columns if col != target and (col in initial_features or any(col.startswith(prefix) for prefix in categorical_features))]

# Prepare the feature matrix (X) and target vector (y)
X = data[initial_features]
y = data[target]
feature_names = X.columns
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate polynomial regression for a single feature
def evaluate_polynomial_regression(feature_index, feature_name,max_degree=10):
    X_feature_train = X_train.iloc[:, feature_index].values.reshape(-1, 1)
    X_feature_test = X_test.iloc[:, feature_index].values.reshape(-1, 1)

    mse = []
    degrees = range(1, max_degree + 1)

    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly_train = poly.fit_transform(X_feature_train)
        X_poly_test = poly.transform(X_feature_test)

        model = LinearRegression()
        model.fit(X_poly_train, y_train)

        y_pred = model.predict(X_poly_test)
        mse.append(mean_squared_error(y_test, y_pred))

    # Plot MSE vs Degree

    plt.figure()
    plt.plot(degrees, mse, marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Feature {feature_index} - Polynomial Degree vs MSE')
    plt.show()


    best_degree = degrees[np.argmin(mse)]
    print(mse[np.argmin(mse)])
    return best_degree, mse
'''
# Evaluate each feature and determine the best polynomial degree
best_degrees = {}
for i in range(X.shape[1]):
    best_degree, mse = evaluate_polynomial_regression(i)
    best_degrees[f'Feature {i}'] = best_degree
    print(f'Best polynomial degree for Feature {i}: {best_degree}')
'''
best_degrees = {}
for i, feature_name in enumerate(feature_names):
    best_degree, mse = evaluate_polynomial_regression(i, feature_name)
    best_degrees[feature_name] = best_degree
    print(f'Best polynomial degree for {feature_name}: {best_degree}')


# Output the best degrees for each feature
print("Best polynomial degrees for each feature:", best_degrees)
