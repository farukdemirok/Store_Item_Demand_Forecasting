# STORE ITEM DEMAND FORECASTING

# WORK PROBLEM:

# # A chain of stores wants a 3-month demand forecast for its 10 different stores and 50 different products.

# DATASET STORY:

# # Informations of 10 different stores and 50 different products that store chain company have are in dataset

# VARIABLES:

# # date: history of sales data
# # store: store ID
# # item: product ID
# # sales: number of products sold

# TASK:

# # Creating 3-month demand forecasting with using time series and machine learning methods.


## >>>><<<<<<<>>>>>><<<<<

########## IMPORT LIBRARIES ############

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# DATA LOADING

train = pd.read_csv('HAFTA_09/Ders Notları/train.csv', parse_dates=['date'])
test = pd.read_csv('HAFTA_09/Ders Notları/test.csv', parse_dates=['date'])

train.shape
test.shape

train.head()
test.head()

train.columns
test.columns

check_df(train)
check_df(test)

# merging 2 dataframe
df = pd.concat([train, test], sort=False)
df.head()

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()
check_df(train)
check_df(test)
check_df(df)


# Satış dağılımı nasıl?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# Store counts
df[["store"]].nunique()

# Item counts
df[["item"]].nunique()

# Are stores counts equal unique item counts?
df.groupby(["store"])["item"].nunique()

# Does stores make same number sales?
df.groupby(["store", "item"]).agg({"sales": ["sum"]}).head()

# sales statistics of store-item breakdown for some example item10
df.groupby(["store",df["item"]==10]).agg({"sales": ["sum","mean", "median", "std"]})


########################
# Date Features
########################

"the reason why we did with below the function is capture the seasonality of values"

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df


df = create_date_features(df)

check_df(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]}).head(10)


########################
# Random Noise
########################

"random noise makes dataset more standart with adding random values"

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag/Shifted Features
########################

"main goal is predict 3-months sales because of that we have to add lag features to capture trend and seasonality." \
"Includes only past values may cause to data leakege. For prevent leakege, we add lag values to dataset"

"we editing dataframe shape like which we want"
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


# creating lag features
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
check_df(df)



########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])
df.tail()


########################
# Exponentially Weighted Mean Features
########################

"we'r trying to capture values different from independent variable and represent trend and seasonality"

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

"when alpha value close to 1, it means give more weight to closest obvervations, in other case that opposite situation " \
"when alpha value close to 0, it means give more weight to distant observations"

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)


########################
# One-Hot Encoding
########################
# creating dummy variable for categoricals

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


########################
# Converting sales to log(1+sales)
########################

"We want to reduce optimization time with using log function which gbm based"

df['sales'] = np.log1p(df["sales"].values)
check_df(df)


#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


########################
# Time-Based Validation Sets
########################
test

# train set till the beginning of 2017
train = df.loc[(df["date"] < "2017-01-01"), :]


# I would choose as a validation set first 3 month of 2017 to predict 2018 first 3 month.
# my basic purpose is make validation set similar to scenario.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# remove independent variables
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]


# we reached out to goal that y validation set similar to test set which in 2018
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape



########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain, # parameters
                  valid_sets=[lgbtrain, lgbval], # datasets
                  num_boost_round=lgb_params['num_boost_round'], # coming from function above
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], # coming from function above
                  feval=lgbm_smape,
                  verbose_eval=100) # give reports per 100 iterations


# model.best_iteration: best iteration
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)


# we made standardization on dataset, we need to take it back.

smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Feature importances
########################
def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))



plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


########################
# Final Model
########################

train = df.loc[~df.sales.isna()] # not na values on sales variable
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()] # na values on sales variable
X_test = test[cols]




lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset

# creating train set again
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)


# model creating
model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


# prediction test dataset
test_preds = model.predict(X_test, num_iteration=model.best_iteration)
