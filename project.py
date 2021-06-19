import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# XGBoost
import xgboost as xgb

# warning 
import warnings
warnings.filterwarnings('ignore')

column_name = ["MPG", "Cylinders", "Displacement","Horsepower","Weight","Acceleration","Model Year", "Origin"]
data = pd.read_csv("auto-mpg.data", names = column_name, na_values = "?", comment = "\t",sep = " ", skipinitialspace = True)

data = data.rename(columns = {"MPG":"target"})

print(data.head())
print("Data Shape: ", data.shape)

data.info()

describe = data.describe()

# %%missing value
print(data.isna().sum())

data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())

print(data.isna().sum())

sns.distplot(data.Horsepower)


# %%EDA
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["target"])>threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()

plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

#box
for c in data.columns:
    plt.figure()
    sns.boxplot(x = c, data = data, orient = "v")


#%%

thr = 2
horsepower_desc = describe["Horsepower"]
q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp
filter_hp_botom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_botom & filter_hp_top

data = data[filter_hp]


acceleration_desc = describe["Acceleration"]
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_botom = bottom_limit_acc < data["Acceleration"]
filter_acc_top = data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_botom & filter_acc_top

data = data[filter_acc]


#%%Feature Engineering
#Skewness

sns.distplot(data.target, fit = norm)

(mu, sigma) = norm.fit(data["target"])
print("mu: {}, sigma = {}".format(mu, sigma))

#qq plot
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()

data["target"] = np.log1p(data["target"])

plt.figure()
sns.distplot(data.target, fit = norm)

(mu, sigma) = norm.fit(data["target"])
print("mu: {}, sigma = {}".format(mu, sigma))

#qq plot
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()


# feature - independent variable
skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = ["skewed"])


#%% one hot encoding
data["Cylinders"] = data["Cylinders"].astype(str)
data["Origin"] = data["Origin"].astype(str)

data = pd.get_dummies(data)


#%% Split - Standardization

#Split
x = data.drop(["target"], axis = 1)
y = data.target

test_size = 0.9
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

#Standardization
scaler = RobustScaler()  #StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%% Regression Models

#Linear regression
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("LR Coef: ", lr.coef_)
y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Linear Regression MSE: ",mse)



#Ridge regression
ridge = Ridge(random_state = 42, max_iter = 10000)

alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha':alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv= n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
score_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ", clf.best_estimator_.coef_)

ridge = clf.best_estimator_

print("Ridge Best Estimator: ", ridge)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Ridge MSE: ", mse)

print("-----------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")



#Lasso regression
lasso = Lasso(random_state = 42, max_iter = 10000)

alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha':alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv= n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)
scores = clf.cv_results_["mean_test_score"]
score_std = clf.cv_results_["std_test_score"]

print("Lasso Coef: ", clf.best_estimator_.coef_)

lasso = clf.best_estimator_

print("Lasso Best Estimator: ", lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Lasso MSE: ", mse)

print("-----------------------")

plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")


#ElasticNet
parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state = 42, max_iter = 10000)
clf = GridSearchCV(eNet, parametersGrid, cv= n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, Y_train)

print("ElasticNet Coef: ", clf.best_estimator_.coef_)


print("ElasticNet Best Estimator: ", clf.best_estimator_)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("ElasticNet MSE: ", mse)



#%%XGBoost
parametersGrid = {'nthread': [4],
                  'objective': ['reg:linear'],
                  'learning_rate':[.03, 0.05, .07],
                  'max_depth':[5, 6, 7],
                  'min_child_weight':[4],
                  'silent': [1],
                  'subsample':[0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv= n_folds, scoring = "neg_mean_squared_error", refit = True, n_jobs=5, verbose=True)
model_xgb = clf.best_estimator_

clf.fit(X_train, Y_train)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("XGBRegressor MSE: ",mse)


#%% Averaging Models

class AveragingModels():
    def __init__(self, models):
        self.models = models
        

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
       
        for model in self.models_:
            model.fit(X, y)

        return self
    
   
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)  


averaged_models = AveragingModels(models = (model_xgb, lasso))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Averaged Models MSE: ",mse)       






















 


