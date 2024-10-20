from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


data = pd.read_csv('four_year_colleges.csv')

y = data['default_rate']
X = data.drop(['default_rate', 'OPEID', 'name'], axis=1)

# Filter out non-numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]
# Split the data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=5, step=1)
selector.fit(X_train, y_train)


print(selector.get_feature_names_out(input_features=None))
from sklearn.preprocessing import SplineTransformer


# Create spline features for all variables on the training set
spline_transformer = SplineTransformer(n_knots=5, degree=2)
X_train_splines = spline_transformer.fit_transform(X_train[selector.get_feature_names_out(input_features=None)])

from sklearn.linear_model import LinearRegression

# Train a model with the spline features on the training set
model = LinearRegression()
model.fit(X_train_splines, y_train)

from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features based on your guess
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_val_poly = poly.transform(X_val)

# Train a model with the polynomial features on the training set
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Evaluate the model on the test set
from sklearn.metrics import r2_score
y_pred = model_poly.predict(X_test_poly)
test_r2 = r2_score(y_test, y_pred)
print(f"Test set R²: {test_r2:.3f}")

# Validate the final model on the validation set
y_val_pred = model_poly.predict(X_val_poly)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation set R²: {val_r2:.3f}")
