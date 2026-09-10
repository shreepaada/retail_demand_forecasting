import pandas as pd 

# load training data 
train = pd.read_csv("data/train.csv")

train["date"] = pd.to_datetime(train["date"])

print("First 5 rows")
print(train.head())
# rows and columns
print("\n Shape")
print(train.shape)

#
print("\ncloumns")
print(train.columns.tolist())

print("\ndata types")
print(train.dtypes)

print("\n Missing Values:")
print(train.isnull().sum())

print("\n Date Range:")
print(train["date"].min(), train["date"].max())

print("\nno of stores")
print(train["store_nbr"].nunique())  #no of unique values in nunique

print("\nproduct family")
print(train["family"].nunique())
print(train["family"].unique())

print("\n sales statistics")
print(train["sales"].describe())

print("\npromotion statistics")
print(train["onpromotion"].describe())
