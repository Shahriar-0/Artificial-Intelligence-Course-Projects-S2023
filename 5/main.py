import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from copy import deepcopy
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import category_encoders as cat_enc
from mlxtend.evaluate import bias_variance_decomp

DATASET_PATH = "assets/house_data.csv"

df = pd.read_csv(DATASET_PATH)
pd.set_option("display.max_columns", None)
def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    nan_values_count = df.isna().sum()
    nan_values_percent = nan_values_count / len(df)
    nan_values = pd.concat(
        [nan_values_count, nan_values_percent], axis=1, keys=["Missing", "Percentage"]
    )
    return nan_values
missing_values(df)
plt.figure(figsize=(20, 20))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".3f",
    cmap="Blues",
    linewidths=1,
    square=True,
)
plt.show()
price_corr = df.corr(numeric_only=True)["price"].drop("price")
price_corr = price_corr[abs(price_corr) > 0.31].sort_values(ascending=False)
print(price_corr)
NUM_OF_INTERVALS = 20
df_backup = deepcopy(df)
sns.set(rc={"figure.figsize": (11.7, 8.27)})
for col in price_corr.index.to_list():
    if df[col].quantile(0.9) - df[col].quantile(0.1) < 20:
        sns.countplot(x=col, data=df)
        plt.show()
    else:
        intervals = pd.interval_range(
            start=df[col].quantile(0.1),
            end=df[col].quantile(0.9),
            periods=NUM_OF_INTERVALS + 1,
        )
        interval_tuples = [(interval.left, interval.right) for interval in intervals]
        bins = pd.IntervalIndex.from_tuples(interval_tuples)
        df[col] = pd.cut(df[col], bins)
        ax = sns.countplot(x=col, data=df)
        ax.set_xticklabels(
            [f"{int(np.mean(interval))}" for interval in interval_tuples]
        )
        plt.show()
df = deepcopy(df_backup)
def plot_corr_scatter_hexbin(col):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    plt.suptitle(col)
    axs[0].scatter(df[col], df["price"])
    axs[1].hexbin(df[col], df["price"], gridsize=20, cmap="Blues")
    plt.subplots_adjust(wspace=0.3)
    plt.show()
for col in price_corr.index:
    plot_corr_scatter_hexbin(col)
def plot_corr_joint(col):
    sns.jointplot(x=col, y="price", data=df, kind="hex")
for col in price_corr.index:
    plot_corr_joint(col)
negative_columns = ["bedrooms", "bathrooms", "sqft_living", "grade"]
df[negative_columns] = np.where(df[negative_columns] < 0, np.nan, df[negative_columns])
missing_values(df)
df.fillna(df.median(numeric_only=True), inplace=True)
missing_values(df)
missing = df_backup[df_backup.isna().sum(axis=1) > 2]
df_imputed = deepcopy(df_backup)
df_imputed.drop(missing.index, inplace=True)
df_imputed = df_imputed.drop(["date", "location", "style"], axis=1)
df_imputed.reset_index(drop=True, inplace=True)
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(df_imputed)
imputed = pd.DataFrame(imputed, columns=df_imputed.columns)
imputed[negative_columns] = np.where(
    imputed[negative_columns] < 0, np.nan, imputed[negative_columns]
)
class DataScaler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include="number")
        self.scaler_std = StandardScaler()
        self.scaler_norm = MinMaxScaler()

    def standardization(self, exclude_cols: list = []):
        self.df[self.numeric_cols.columns] = self.scaler_std.fit_transform(
            self.numeric_cols
        )
        self.df[exclude_cols] = self.numeric_cols[exclude_cols]

    def normalization(self, exclude_cols: list = []):
        self.df[self.numeric_cols.columns] = self.scaler_norm.fit_transform(
            self.numeric_cols
        )
        self.df[exclude_cols] = self.numeric_cols[exclude_cols]
scalar = DataScaler(df)
scalar.df.hist(bins=20, figsize=(20, 15))
plt.show()
scalar.normalization(exclude_cols=["price"])
scalar.df.hist(bins=20, figsize=(20, 15))
plt.show()
class CategoricalEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cat_cols = df.select_dtypes(include=["category", "object"])

        self.encoders = {
            "label": cat_enc.OrdinalEncoder(cols=self.cat_cols.columns),
            "one-hot": cat_enc.OneHotEncoder(
                cols=self.cat_cols.columns, use_cat_names=True
            ),
            "target": cat_enc.TargetEncoder(
                cols=self.cat_cols.columns, min_samples_leaf=2, smoothing=1.1
            ),
            "frequency": cat_enc.CountEncoder(cols=self.cat_cols.columns),
            "binary": cat_enc.BinaryEncoder(cols=self.cat_cols.columns),
        }

    def encode(self, mode: str, target: str = None):
        if mode != "target":
            self.df[self.cat_cols.columns] = self.encoders[mode].fit_transform(
                self.cat_cols
            )
        else:
            self.df[self.cat_cols.columns] = self.encoders[mode].fit_transform(
                self.cat_cols, self.df[target]
            )
encoder = CategoricalEncoder(df)
encoder.encode("label")
df = df[price_corr.index.union(["price"])]
class DataSplitter:
    def __init__(self, df: pd.DataFrame, train_percent: float = 0.8):
        self.data = df[df.columns.difference(["price"])]
        self.outcome_data = df["price"]
        self.__split(train_percent)

    def __split(self, train_percent: float):
        train_feat, test_feat, train_out, test_out = train_test_split(
            self.data, self.outcome_data, train_size=train_percent, random_state=1
        )
        self.data_train = train_feat
        self.data_test = test_feat
        self.outcome_train = train_out
        self.outcome_test = test_out
        
dataSplitter = DataSplitter(df)

def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    input_sum = np.sum(input_feature)
    output_sum = np.sum(output)

    # compute the product of the output and the input_feature and its sum
    product_sum = np.sum(output * input_feature)

    # compute the squared value of the input_feature and its sum
    input_squared_sum = np.sum(input_feature ** 2)

    # use the formula for the slope
    slope = (product_sum - (input_sum * output_sum) / len(input_feature)) / (input_squared_sum - (input_sum ** 2) / len(input_feature))

    # use the formula for the intercept
    intercept = output_sum / len(input_feature) - slope * input_sum / len(input_feature)

    return (intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + slope * input_feature

    return predicted_values

# def simple_linear_regression(input_feature, output):
#     # create a matrix of input features with an additional column of ones for the intercept
#     X = np.vstack((input_feature, np.ones(len(input_feature)))).T

#     # use np.linalg to solve for the coefficients
#     coefficients = np.linalg.lstsq(X, output, rcond=None)[0]

#     # extract the slope and intercept from the coefficients
#     slope = coefficients[0]
#     intercept = coefficients[1]

#     return (intercept, slope)

# def get_regression_predictions(input_feature, intercept, slope):
#     # create a matrix of input features with an additional column of ones for the intercept
#     X = np.vstack((input_feature, np.ones(len(input_feature)))).T

#     # use the coefficients to make predictions
#     predicted_values = np.dot(X, [slope, intercept])

#     return predicted_values


def get_root_mean_square_error(predicted_values, output):
    # Compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = output - predicted_values

    # square the residuals and add them up
    residuals_squared_sum = np.sum(residuals ** 2)

    # find the mean of the above phrase
    mean_residuals_squared = residuals_squared_sum / len(output)

    # calculate the root
    RMSE = np.sqrt(mean_residuals_squared)

    return RMSE

def get_r2_score(predicted_values, output):
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = output - predicted_values

    # square the residuals and add them up -> SSres
    SSres = np.sum(residuals ** 2)

    # compute the SStot
    SStot = np.sum((output - np.mean(output)) ** 2)

    # compute the R2 score value
    R2_score = 1 - SSres / SStot

    return R2_score

# TO DO:

designated_feature_list = ["sqft_living", "yr_built", "grade", "zipcode"]

input_feature = np.array([1, 2, 3, 4, 5])
output = np.array([1, 3, 2, 5, 7])

for feature in designated_feature_list:
    # calculate R2 score and RMSE for each given feature
    intercept, slope = simple_linear_regression(input_feature, output)
    predicted_values = get_regression_predictions(input_feature, intercept, slope)
    RMSE = get_root_mean_square_error(predicted_values, output)
    R2_score = get_r2_score(predicted_values, output)

    print("Feature:", feature)
    print("Intercept:", intercept)
    print("Slope:", slope)
    print("RMSE:", RMSE)
    print("R2 Score:", R2_score)