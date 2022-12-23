# -*- coding: utf-8 -*-
#%% Librairies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn.inspection import permutation_importance

#%% Path of the dataset

# Path of the dataset

path_75 = 'df_europewinds_75.pkl'
path_95 = 'df_europewinds_95.pkl'

data_75 = pd.read_pickle(open(path_75, 'rb'))
describ_75 = data_75.describe()

data_95 = pd.read_pickle(open(path_95, 'rb'))
describ_95 = data_95.describe()

# Intensity 1 means extreme convective winds bursts and 0 means moderate convective bursts

data_75['intensity']=0
data_95['intensity']=1

# Merge the 2 dataset and separate the features from the intensity

data_merged = pd.concat((data_75,data_95))
y_all = data_merged['intensity']
X_all = data_merged.iloc[:,2:-1]

# Store the name of each feature

features_names = data_merged.columns[2:-1]

#%% Dataset 

#%%% Splitting of the dataset with years
'''
years_training = np.arange(0,7)
years_test = np.arange(7,8)
years_valid = np.arange(8,9)

data_training = data_merged.loc[data_merged['year'].isin(years_training)]
data_test = data_merged.loc[data_merged['year'].isin(years_test)]
data_valid = data_merged.loc[data_merged['year'].isin(years_valid)]

X_train = data_training.iloc[:,2:-1]
y_train = data_training['intensity']
X_test = data_test.iloc[:,2:-1]
y_test = data_test['intensity']
X_valid = data_valid.iloc[:,2:-1]
y_valid = data_valid['intensity']

'''
#%%% Splitting the dataset randomly 

# Split the dataset into a training (64%), validation (16%), and testing set (20%)

X_train_valid, X_test, y_train_valid, y_test = train_test_split(X_all, y_all, train_size=0.8,random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, train_size=0.8,random_state=42)

#%% Logistic Regression 
#%%% with default settings
# Let's try to predict the 2 classes of winds bursts with Logistic Regression  

lr_default = LogisticRegression(random_state=42)
lr_default.fit(X_train, y_train)

# Let's try a prediction with the validation set

prediction_lr_default = lr_default.predict(X_valid)

lr_default_accuracy = accuracy_score(y_valid, prediction_lr_default)

print(f"Accuracy of the default Logistic Regression over the validation set: {lr_default_accuracy:.2%}")

# The accuracy is ok, around 58%, next the precision score :

lr_default_precision = precision_score(y_valid, prediction_lr_default)

print(f"Precision of the default Logistic Regression over the validation set: {lr_default_precision:.2%}")

# 52%, not great, so let's try to adujst the solver


#%%% with costums settings

lr_custom = LogisticRegression(solver='liblinear',
                                random_state=42)
lr_custom.fit(X_train, y_train)

# Let's try a prediction with the validation set

y_pred_valid_LR = lr_custom.predict(X_valid)

valid_acc_LR = accuracy_score(y_valid, y_pred_valid_LR)

print(f"Accuracy of the custom Logistic Regression over the validation set: {valid_acc_LR:.2%}")

valid_pre_LR = precision_score(y_valid, y_pred_valid_LR)

print(f"Precision of the custom Logistic Regression over the validation set: {valid_acc_LR:.2%}")

# Now let's compare it with prediction based on the testing set

y_pred_test_LR = lr_custom.predict(X_test)

test_acc_LR = (accuracy_score(y_test,y_pred_test_LR))

print( f'Accuracy of the custom Logistic Regression over the testing set: {test_acc_LR:.2%}')

test_pre_LR = (precision_score(y_test,y_pred_test_LR))

print( f'Precision of the custom Logistic Regression over the testing set: {test_pre_LR:.2%}')

# The accuracy is still not good, just around 60%, eventhough the precision is worst on the test set. 
# Let's see the permutation feature importance onm the custom Logisitic Regression

#%%% Permutation features
# Let's see what features are important for the Logistic Regression
# The following cell was copied from https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# This cell mainly extract the importances of each feature and randomly shuffle them
# It also check how much time it takes

start_time = time.time()
result_LR = permutation_importance(
    lr_custom, X_test, y_test, n_repeats=20, random_state=42, n_jobs=6
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

logistic_importances = pd.Series(result_LR.importances_mean, index=features_names)

# Let's see the results. The higher the bar is, the more important the feature is

fig, ax = plt.subplots()
logistic_importances.plot.bar(yerr=result_LR.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on costum LR")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

#%% Random Forest
#%%% with default settings
# Now let's try with RandomForest

rf_default = RandomForestClassifier(n_jobs = 6, random_state=42)
rf_default.fit(X_train,y_train)

# Let's see if the predictions are better than the Logistic Regression

prediction_rf_default = rf_default.predict(X_valid)
rf_default_accuracy = accuracy_score(y_valid, prediction_rf_default)

print(f"Accuracy of the default Random Forest over the validation set: {rf_default_accuracy:.2%}")

# 67% which is better than the Logistic Regression

rf_default_precision = precision_score(y_valid, prediction_rf_default)

print(f"Precision of the default Random Forest over the validation set: {rf_default_precision:.2%}")

# Better from the first try, let's try to improve it

#%%% with hyperparameters tuning 

# Tune of the hyperparameters for the Random Forest with HalvingGridSearchCV
# param_grid helps to define the hyperparameters you would like to test, 
# and the ranges the hyperparameters should be in.
# To avoid long computing times, here are the best parameters I've found:

rf_custom = RandomForestClassifier(max_leaf_nodes=200,
                                   min_samples_split=15,
                                   n_estimators=150,
                                   max_depth= 10,
                                   random_state=42,
                                   n_jobs=10)

# Next, fit with the training data and customs parameters:

start_time = time.time()
rf_custom.fit(X_train,y_train)
elapsed_time_RF = time.time() - start_time
print(f"Elapsed time to fit the training set with tuned hyperparameters of Random Forest: {elapsed_time_RF:.3f} seconds")

#%%% The tuned Random Forest

# Let's see the prediction from the tuned RF

y_pred_valid_RF = rf_custom.predict(X_valid)
valid_acc_RF = accuracy_score(y_valid, y_pred_valid_RF)

print(f"Accuracy of the custom Random Forest over the validation set {valid_acc_RF:.2%}")

valid_pre_RF = precision_score(y_valid, y_pred_valid_RF)

print(f"Precision of the custom Random Forest over the validation set {valid_pre_RF:.2%}")

# Now let's compare it with prediction based on the testing set

y_pred_test_RF = rf_custom.predict(X_test)

test_acc_RF = (accuracy_score(y_test,y_pred_test_RF))

print(f'Accuracy of the custom Random Forest over the testing set: {test_acc_RF:.2%} \n')

test_pre_RF = (precision_score(y_test,y_pred_test_RF))

print(f'Precision of the custom Random Forest over the testing set: {test_pre_RF:.2%} \n')

# Results are slightly worst than the default Random Forest (66 % instead of 67 %), but performs best on the testing set

#%%% Permutation features

# Since the performance are not clearly better than the Logistic Regression,
# let's see what features are important for the default Random Forest

start_time = time.time()
result_RF = permutation_importance(
    rf_default, X_test, y_test, n_repeats=20, random_state=42, n_jobs=6
)

elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances of the Random Forest: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result_RF.importances_mean, index=features_names)

# Let's see the resutls. The higher the bar is, the more important the feature is

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result_RF.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on the default Random Forest")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

#%% Decision Tree
#%%% with default parameters

dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train,y_train)

# Let's see if the predictions are better than the others

prediction_dt_default = dt_default.predict(X_valid)
dt_default_accuracy = accuracy_score(y_valid, prediction_dt_default)

print(f"Accuracy of the default Decision Tree over the validation set: {dt_default_accuracy:.2%}")

dt_default_precision = precision_score(y_valid, prediction_dt_default)

print(f"Precision of the default Decision Tree over the validation set: {dt_default_precision:.2%}")

# Results are in the same line as the Logisitic Regression, let's try to improve it

#%%% with hyperparameters tuning

# A tuning with HalvingGridSearchCV was also done, but to avoid long 
# computing times, here are the best parameters I've found:

dt_custom = DecisionTreeClassifier(max_leaf_nodes=60,
                                   min_samples_split=20,
                                   max_depth= 10,
                                   random_state=42)

# Next, try a new fit with the training data

start_time = time.time()
dt_custom.fit(X_train,y_train)
elapsed_time_DT = time.time() - start_time

print(f"Elapsed time to fit the training set with tuned hyperparameters of Decision Tree: {elapsed_time_DT:.3f} seconds")

#%%% The tuned Decision Tree

# Let's see the prediction from the tuned RF

y_pred_valid_DT = dt_custom.predict(X_valid)
valid_acc_DT = accuracy_score(y_valid, y_pred_valid_DT)

print(f"Accuracy of the custom Decision Tree over the validation set: {valid_acc_DT:.2%}")

valid_pre_DT = precision_score(y_valid, y_pred_valid_DT)

print(f"Precision of the custom Decision Tree over the validation set: {valid_pre_DT:.2%}")

# Now let's compare it with prediction based on the testing set

y_pred_test_DT = dt_custom.predict(X_test)

test_acc_DT = (accuracy_score(y_test,y_pred_test_DT))

print(f'Accuracy of the custom Decision Tree over the testing set: {test_acc_DT:.2%} \n')

test_pre_DT = (precision_score(y_test,y_pred_test_DT))

print(f'Precision of the custom Decision Tree over the testing set: {test_pre_DT:.2%} \n')

# Results are better than Logistic Regression but worst than Random Forest 

#%%% Permutation features

start_time = time.time()
result_dt = permutation_importance(
    dt_custom, X_test, y_test, n_repeats=20, random_state=42, n_jobs=6
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

decision_tree_importances = pd.Series(result_dt.importances_mean, index=features_names)

# Let's see the results. The higher the bar is, the more important the feature is

fig, ax = plt.subplots()
decision_tree_importances.plot.bar(yerr=result_dt.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on costum DT")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

#%% Boosted Gradients
#%%% with default settings

gb_default = GradientBoostingClassifier(random_state=42)
gb_default.fit(X_train, y_train)

prediction_gb_default = gb_default.predict(X_valid)
gb_default_accuracy = accuracy_score(y_valid, prediction_gb_default)

print(f"Accuracy of the default Gradient Boosting over the validation set: {gb_default_accuracy:.2%}")

gb_default_precision = precision_score(y_valid, prediction_gb_default)

print(f"Precision of the default Gradient Boosting over the validation set: {gb_default_precision:.2%}")

# Same results as custom Random Forest

#%%% with hyperparameters tuning

# Best parameters found so far:

start_time = time.time()

gb_custom = GradientBoostingClassifier(max_leaf_nodes=180,
                                       min_samples_split=25,
                                       learning_rate=0.2,
                                       max_depth= 3,
                                       random_state=42)
gb_custom.fit(X_train,y_train)
elapsed_time_GB = time.time() - start_time
print(f"Elapsed time to fit the training set with tuned hyperparameters of Gradient Boosting : {elapsed_time_GB:.3f} seconds")

#%%% The tuned Gradient Boosting

# Let's see the prediction from the tuned RF

y_pred_valid_GB = gb_custom.predict(X_valid)
valid_acc_GB = accuracy_score(y_valid, y_pred_valid_GB)

print(f"Accuracy of the custom Gradient Boosting over the validation set: {valid_acc_GB:.2%}")

valid_pre_GB = precision_score(y_valid, y_pred_valid_GB)

print(f"Precision of the custom Gradient Boosting over the validation set: {valid_pre_GB:.2%}")

# Now let's compare it with prediction based on the testing set

y_pred_test_GB = gb_custom.predict(X_test)

test_acc_GB =(accuracy_score(y_test,y_pred_test_GB))

print(f'Accuracy of the custom Gradient Boosting over the testing set: {test_acc_GB:.2%} \n')

test_pre_GB =(precision_score(y_test,y_pred_test_GB))

print(f'Precision of the custom Gradient Boosting over the testing set: {test_pre_GB:.2%} \n')


# Results are marginally better than the default run, but 
# the testing set is worst than the default run

#%%% Permutation feature

start_time = time.time()
result_gb = permutation_importance(
                                   gb_custom, X_test, y_test, n_repeats=20, random_state=42, n_jobs=6
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

gradient_boosting_importances = pd.Series(result_gb.importances_mean, index=features_names)

# Let's see the resutls. The higher the bar is, the more important the feature is

fig, ax = plt.subplots()
gradient_boosting_importances.plot.bar(yerr=result_gb.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on costum GB")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

#%% Resume of accuracies 

# This cell resumes the accuracies of each default and custom algorithm

import matplotlib.ticker as mtick

accuracies_custom = pd.Series([
                              valid_acc_LR,
                              valid_acc_RF,
                              valid_acc_DT,
                              valid_acc_GB
                              ])
accuracies_default = pd.Series([
                                lr_default_accuracy,
                                rf_default_accuracy,
                                dt_default_accuracy,
                                gb_default_accuracy,])
algo = ['LR',
        'LR 2',
        'RF',
        'RF 2',
        'DT',
        'DT 2',
        'GB',
        'GB 2']
length_algo = np.arange(4)
width = 0.2

fig, ax = plt.subplots(1,figsize=(12,9))
ax.bar(length_algo + 0.5*width, accuracies_default*100, width)
ax.bar(length_algo + 1.5*width, accuracies_custom*100, width)
ax.set_title('Accuracies score of all the models tested on the validation set')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0,100)
plt.xticks(length_algo+width,['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting' ])
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid(which='minor', axis='y',linestyle='dashed')
ax.grid(which='major', axis='y', linestyle='-')
plt.legend(['Default settings','Custom settings'])
fig.savefig('accuracies.png', dpi=300)

#%% Resume of precision

# This cell resumes the precision of each default and custom algorithm

precision_default_valid = pd.Series([precision_score(y_valid,prediction_lr_default),
                                     precision_score(y_valid,prediction_rf_default),
                                     precision_score(y_valid,prediction_dt_default),
                                     precision_score(y_valid,prediction_gb_default),
                                     ])
precision_custom_valid = pd.Series([precision_score(y_valid,y_pred_valid_LR),
                             precision_score(y_valid,y_pred_valid_RF),
                             precision_score(y_valid,y_pred_valid_DT),
                             precision_score(y_valid,y_pred_valid_GB)])

fig, ax2 = plt.subplots(1,figsize=(12,9))
ax2.bar(length_algo + 0.5*width, precision_default_valid*100, width)
ax2.bar(length_algo + 1.5*width, precision_custom_valid*100, width)
ax2.set_title('Precision score of all the models tested on the validation set')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylim(0,100)
plt.xticks(length_algo+width,['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting' ])
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.grid(which='minor', axis='y',linestyle='dashed')
ax2.grid(which='major', axis='y', linestyle='-')
plt.legend(['Default settings','Custom settings'])
fig.savefig('precision.png', dpi=300)

#%% Feature importances on all models

# This cell plots the most important features of each algorithm (the best
# algorithm between the default and the custom one)

length_names = np.arange(len(features_names))
width = 0.2;

fig2, ax2 = plt.subplots(1, figsize=(20,8))
ax2.bar(length_names + 0*width, logistic_importances,width)
ax2.bar(length_names + 1*width, forest_importances, width)
ax2.bar(length_names + 2*width, decision_tree_importances, width)
ax2.bar(length_names + 3*width, gradient_boosting_importances, width)
plt.xticks(length_names+width,features_names)
plt.xticks(rotation=45)
plt.legend(['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting' ])
ax2.set_title("Feature importances using feature permutation on all models tested")
ax2.set_ylabel("Mean accuracy decrease")
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.set_xlabel('Predictors')
ax2.minorticks_on()
ax2.grid(which='major', axis='y' ,color='gray', linestyle='dashed')
ax2.grid(which='minor', axis='y', linestyle=':')
fig2.tight_layout()
fig2.show()
fig2.savefig('feat_import_all.png', dpi=300)