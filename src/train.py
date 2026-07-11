import pandas as pd 
df=pd.read_csv("data/train (1).csv")

# print("\n firest 5 rows:")
# print(df.head())

# print("\n shape of dataset:")
# print(df.shape)

# print("\n columns:")
# print(df.columns)

# print("\n dataset information:")
# print(df.info())

# print("\n missing values:")
# print(df.isnull().sum())

# print("\nstatistical summary:")
# print(df.describe())


#import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 5))
# plt.hist(df["SalePrice"],bins=30)
# plt.title("Distribution of house price")
# plt.xlabel("Sale price")
# plt.ylabel("Number of Houses")
# plt.show()


#scatter plot:living area vs sale price
# plt.figure(figsize=(8, 5))
# plt.scatter(df["GrLivArea"],df["SalePrice"])
# plt.title("Living Area vs Sale Price")
# plt.xlabel("Ground Living Area(sq ft)")
# plt.ylabel("Sale Price")
# plt.show()

#correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
 
#select only numerical columns
numeric_df= df.select_dtypes(include=['int64','float64'])

#calculate correlation 
correlation=numeric_df.corr()

#plot heatmap
# plt.figure(figsize=(12,10))
# sns.heatmap(correlation,cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

# saleprice_corr=correlation["SalePrice"].sort_values(ascending=False)
# print(saleprice_corr)

#select features
X=df[[
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath"
]]

# target
Y=df["SalePrice"]

print("Feature (X)")
print(X.head())

print("\nTarget (Y)")
print(Y.head())


from sklearn.model_selection import train_test_split
# split dataset
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=56
)
print("training data:",X_train.shape)
print("testing data:",X_test.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#create linear reg model
model =LinearRegression()

# train models
model.fit(X_train,Y_train)
print("Model trained successfully")

# predict house prices
Y_pred=model.predict(X_test)
print("first 10 predication:")
print(Y_pred[:10])

#Compare actual vs predicted
comparision=pd.DataFrame({
    "Actual price":Y_test.values,
    "Predicted Price":Y_pred
})
print(comparision.head(10))

#evaluate model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae=mean_absolute_error(Y_test,Y_pred)
mse=mean_squared_error(Y_test,Y_pred)
r2=r2_score(Y_test,Y_pred)

print("\n Model Evalution")
print(f"MAE:{mae:.2f}")
print(f"MAE:{mse:.2f}")
print(f"R2 Score:{r2:.4f}")