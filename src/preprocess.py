import pandas as pd

#load data set
df=pd.read_csv("data/train (1).csv")

#checking missing values
missing=df.isnull().sum()
print("Missing values:")
print(missing[missing>0].sort_values(ascending=False))

#drop the clumes that have too many missing values
columns_to_drop=[
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence"
]

df.drop(columns=columns_to_drop,inplace=True)

# Fill numerical columns
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())

# Fill categorical columns
categorical_cols = [
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
    "Electrical",
    "FireplaceQu"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after preprocessing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

#Convert categorical coulmns into numrical columns  one-hot encoding
df_encoded=pd.get_dummies(df,drop_first=True)

#display the first 5 rows
print(df_encoded.head())

#check the new shape
print("\n Shap after encoding:")
print(df_encoded.shape)

print(df_encoded.columns)

#separate the datas
X = df_encoded.drop("SalePrice", axis=1)
y = df_encoded["SalePrice"]

#split data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#make evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)