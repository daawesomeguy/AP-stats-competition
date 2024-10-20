import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('four_year_colleges.csv')

# Define the target variable and initial set of features
target = 'default_rate'
initial_features = [
    'median_debt', 'highest_degree', 'ownership', 'locale', 'admit_rate', 'SAT_avg',
    'online_only', 'enrollment', 'net_price', 'avg_cost', 'net_tuition', 'ed_spending_per_student',
    'avg_faculty_salary', 'pct_PELL', 'pct_fed_loan', 'grad_rate', 'pct_firstgen', 
    'med_fam_income', 'med_alum_earnings'
]

# Convert categorical features to numeric if they exist in the dataset
categorical_features = ['highest_degree', 'ownership', 'locale', 'online_only']
for feature in categorical_features:
    if feature in data.columns:
        data = pd.get_dummies(data, columns=[feature], drop_first=True)

# Update initial_features after dummification
initial_features = [col for col in data.columns if col != target and (col in initial_features or any(col.startswith(prefix) for prefix in categorical_features))]

# Prepare the feature matrix (X) and target vector (y)
X = data[initial_features]
y = data[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Correlation analysis to select initially relevant features
corr = pd.Series(np.abs(np.corrcoef(X_train, y_train, rowvar=False)[-1, :-1]), index=X.columns)
initial_features_corr = corr[corr > 0.25].index  # Adjust the correlation threshold as needed

# Ensure the indices are correctly converted to integers
initial_features_corr_indices = [X.columns.get_loc(feature) for feature in initial_features_corr]

# Recursive Feature Elimination (RFE) with Linear Regression
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=8, step=1)
selector.fit(X_train[:, initial_features_corr_indices], y_train)
linear_features = initial_features_corr[selector.support_]
print(linear_features)
# Polynomial degree parameter
polynomial_degree = 4  # You can set this to any integer value

# Polynomial Features Creation
poly_features = []
for i, max_degree in enumerate([3, 2]):
    poly = PolynomialFeatures(degree=max_degree)
    poly_features.append(poly.fit_transform(X[:, [i]]))
#poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
X_poly_train = np.column_stack(poly_features)
X_poly_test = np.column_stack(poly_features)
#X_poly_train = poly.fit_transform(X_train[:, initial_features_corr_indices][..., selector.support_])
#X_poly_test = poly.transform(X_test[:, initial_features_corr_indices][..., selector.support_])

# Define the Lasso model with the given alpha
lasso_alpha = 0.03359818286283781
lasso = Lasso(alpha=lasso_alpha, random_state=42, max_iter=1000000)
lasso.fit(X_poly_train, y_train)

# Get the feature importances from the Lasso model
feature_importances = lasso.coef_

# Select the indices of the non-zero coefficients
selected_feature_indices = [i for i, coef in enumerate(feature_importances) if coef != 0]

X_poly_train_selected = X_poly_train[:, selected_feature_indices]
X_poly_test_selected = X_poly_test[:, selected_feature_indices]

# Train the Lasso model on selected polynomial features
lasso.fit(X_poly_train_selected, y_train)

# Evaluate the Lasso model
y_pred_lasso = lasso.predict(X_poly_test_selected)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso Alpha: {lasso.alpha}")
print(f"Lasso Mean Squared Error: {mse_lasso}")
print(f"Lasso R^2 Score: {r2_lasso}")

# Additional Cross-Validation for Lasso model
cross_val_scores_lasso = cross_val_score(lasso, X_poly_train_selected, y_train, cv=5)
print(f"Lasso Cross-Validation Scores: {cross_val_scores_lasso}")
print(f"Lasso Mean Cross-Validation Score: {np.mean(cross_val_scores_lasso)}")
