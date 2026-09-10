import pandas as pd 
import matplotlib.pyplot as plt

# load data

train = pd.read_csv("data/train.csv")

# convert date

train["date"] = pd.to_datetime(train["date"])

# total daily sales

daily_sales = train.groupby("date")["sales"].sum()

plt.figure(figsize = (14,5))
plt.plot(daily_sales)
plt.title("total daily sales")
plt.xlabel("date")
plt.ylabel("sales")
plt.tight_layout()
plt.show()

# monthly sales


monthly_sales = (train.set_index("date").resample("ME")["sales"].sum())
plt.figure(figsize=(14, 5))
plt.plot(monthly_sales)
plt.title("Total Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# sales by product family


family_sales = train.groupby("family")["sales"].sum().sort_values(ascending=False)
plt.figure(figsize =(12,7))
family_sales.plot(kind = "bar")
plt.title("total sales by product family")
plt.xlabel("product family")
plt.ylabel("total sales")
plt.xticks(rotation = 90)
plt.tight_layout
plt.show()

# sales by stores

store_sales = train.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)

plt.figure(figsize = (12,5))
store_sales.plot(kind = "bar")
plt.title("total sales by store")
plt.xlabel("store")
plt.ylabel("total Sales")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Average sales by day of week
# --------------------------------------------------

train["day_of_week"] = train["date"].dt.dayofweek

dow_sales = (
    train.groupby("day_of_week")["sales"]
    .mean()
)

day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

dow_sales.index = [day_names[i] for i in dow_sales.index]

plt.figure(figsize=(10, 5))
dow_sales.plot(kind="bar")
plt.title("Average Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Sales vs Promotion
# --------------------------------------------------

promotion_sales = (
    train.groupby("onpromotion")["sales"]
    .mean()
)

plt.figure(figsize=(10, 5))
plt.plot(promotion_sales.index, promotion_sales.values)
plt.title("Average Sales vs Number of Products on Promotion")
plt.xlabel("Number of Products on Promotion")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.show()

promotion_bins = pd.cut(
    train["onpromotion"],
    bins=[-1, 0, 5, 10, 20, 50, 100, 200, 1000],
    labels=[
        "0",
        "1-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101-200",
        "201+"
    ]
)

promotion_sales = (
    train.groupby(promotion_bins, observed=True)["sales"]
    .mean()
)

plt.figure(figsize=(10, 5))
promotion_sales.plot(kind="bar")
plt.title("Average Sales by Promotion Level")
plt.xlabel("Products on Promotion")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()