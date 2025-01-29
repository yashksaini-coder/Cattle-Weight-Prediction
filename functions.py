# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df.columns = [col.strip() for col in df.columns]
    df = df.drop(['Campaigns', 'ids', 'Note', 'nSHD'], axis=1, errors='ignore')
    return df

def preprocess_data(df):
    df['Time'] = df['Time'].astype(str).str.replace('PM', '').str.replace('AM', '')
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Time'] = df['Time'].fillna(df['Time'].median())
    df['Usable'] = df['Usable'].fillna(df['Usable'].mode())
    df['Usable'] = df['Usable'].map({'Y': 1, 'N': 0})
    
    label_encoder = LabelEncoder()
    df['Pens'] = label_encoder.fit_transform(df['Pens'])
    
    categorical_features = ['pen']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['Weight']]
    
    return df, categorical_features, numerical_features

def create_features(df):
    df['time_of_day'] = pd.cut(df['Time'], 
                              bins=[0, 6, 12, 18, 24],
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['time_of_day'] = df['time_of_day'].cat.codes
    df['area_ratio'] = df['Area'] / df['Larea']
    return df

def build_model_pipeline(categorical_features, numerical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
        elif df[col].dtype.name == 'category':  
            df[col] = df[col].cat.codes
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
    return df

def transform_features(X, preprocessor):
    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    return X_transformed_df
