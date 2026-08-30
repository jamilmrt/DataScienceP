
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import  LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score



# Load the datasets
housing = pd.read_csv('housing.csv')

housing['income_cat'] = pd.cut(housing['median_income'], bins=[0.0,1.5,3.0,4.5,6.0, np.inf], labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in split.split(housing, housing['income_cat']):
    start_train_set = housing.iloc[train_index].drop('income_cat', axis=1)
    start_test_set = housing.iloc[test_index].drop('income_cat', axis=1)

# we will work on the training data

housing = start_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


# 5 Lets make the pipeline for numerical and categoricals

# for Numericals
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

# for categorical
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
# 6 Transform the data

housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)

# 7 Training Model

# Linear Regressor Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds,)
lin_rmse = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

print(f"The Root Mean Square Error for Linear Regression is {pd.Series(lin_rmse).describe()}")

# Decision Tree Regressor Model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds,)
dec_preds = dec_reg.predict(housing_prepared)

dec_tree_rmse = cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# print(f"The Root Mean Square Error for Decision Tree Regression is {dec_tree_rmse}")
print(f"The Root Mean Square Error for Decision Tree Regression is{pd.Series(dec_tree_rmse).describe()}")

# random Forest Regressor Model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
# random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds,)
random_forest_rmse = cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

print(f"The Root Mean Square Error for Random forest Regression is {pd.Series(random_forest_rmse).describe()}")


























