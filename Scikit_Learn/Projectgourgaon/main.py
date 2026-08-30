import os 
import joblib
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


MODEL_FILE = "model.pkl"
PIPPELINE_FILE = "pipeline.pkl"

#Load Dataset
housing = pd.read_csv("housing.csv")


# create stratified test set3
housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins=[0.0,1.5,3.0,4.5,6.0, np.inf], 
                               labels=[1,2,3,4,5])

def build_pipeline(num_attribs, cat_attribs):
    # for Numercal
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # For categoricals 
    cat_pipeline = Pipeline([
        ("onehor", OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Load the datasets
    housing = pd.read_csv('housing.csv')
    
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0.0,1.5,3.0,4.5,6.0, np.inf], labels=[1,2,3,4,5])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing = housing.iloc[train_index].drop('income_cat', axis=1)

        # 3. Separate predictors and labels
    housing_labels = housing["median_house_value"].copy()
    # housing_features = housing_features.drop("median_house_value", axis=1)

    num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    print(housing_prepared)

        