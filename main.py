import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import pickle
from sklearn.impute import SimpleImputer


from functions import load_data, preprocess_data, create_features, build_model_pipeline, convert_to_numeric, transform_features

df=load_data(file_path='./dataset.xlsx')

sns.set(style="whitegrid")
summary_stats = df.describe()
missing_values = df.isnull().sum()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(20, 15))
df[num_cols].hist(bins=20, figsize=(20, 15), layout=(8, 7), edgecolor='black')
plt.tight_layout()
plt.show()

summary_stats, missing_values[missing_values > 0]

num_features = len(num_cols)
rows = int(np.ceil(num_features / 4))  # 4 columns per row

fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], bins=20, kde=True, ax=axes[i], color='blue')
    axes[i].set_title(f"Distribution of {col}")
for i in range(num_features, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

summary_stats, missing_values[missing_values > 0]

categorical_features = ['Pens']
build_model_pipeline(categorical_features, ['Time', 'Usable', 'Area', 'Larea', 'area_ratio', 'time_of_day'])
df, categorical_features, numerical_features = preprocess_data(df)

df = convert_to_numeric(df)
print(df.dtypes)
df = create_features(df)

# Define the preprocessor
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

# Final feature selection
X = df.drop('Weight', axis=1)
y = df['Weight']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.info(), 

X_test.info()


# Example usage
X_transformed = transform_features(X, preprocessor)
X_transformed.head()

y_train.info(), y_test.info()

model = build_model_pipeline(categorical_features, numerical_features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

categorical_features = ['Pens'] 
numerical_features=['Time', 'Usable', 'Area', 'Larea', 'area_ratio']
# Preprocessing transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])



# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'R² Score': r2, 'RMSE': rmse, 'MAE': mae}
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, label='Predicted vs Actual', color='blue')
    sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal Fit')
    plt.xlabel('Actual Weight')
    plt.ylabel('Predicted Weight')
    plt.title(f'Actual vs Predicted Weights - {name}')
    plt.legend()
    plt.grid(True)
    plt.show()

results_df = pd.DataFrame(results).T
print(results_df)


ridge = Ridge()

# hyperparameter grid
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]
}

imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1,error_score=np.nan)
grid_search.fit(X_train_imputed, y_train)
best_alpha = grid_search.best_params_["alpha"]
print(f"Best Alpha: {best_alpha}")

tuned_rigde_model = Ridge(alpha=best_alpha)
tuned_rigde_model.fit(X_train_imputed, y_train)
y_pred = tuned_rigde_model.predict(X_test_imputed)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Tuned Ridge Regression -> RMSE: {rmse:.4f}, R² Score: {r2:.4f}, MAE: {mae:.4f}")


# Save the model to a file
model_file = open('tuned_ridge_model.pkl', 'wb')
pickle.dump(tuned_rigde_model, model_file)
