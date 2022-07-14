from multiprocessing import Process
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import matplotlib.font_manager

import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTENC

def model(combination):
    y = combination[0]["oa_probility"]
    X=combination[0].loc[:, (combination[0].columns != "oa_probility") ].copy()         
    
    if combination[1]=="x_hybrid": 
        numerical_columns = ["age", "COAUTHOR_COUNT_ITM", "international_coauthor_percent","JOURNAL_RANK2011_2016", "OA_AGIANST_CA_CITE","OA_against_CA_PUBLISH","income_level","APC_USD"]
        categorical_columns = ["FIELD","GENDER", "Springer_agreement"]
    else:
        numerical_columns = ["age", "COAUTHOR_COUNT_ITM", "international_coauthor_percent","JOURNAL_RANK2011_2016", "OA_AGIANST_CA_CITE","OA_against_CA_PUBLISH","income_level"]
        categorical_columns = ["FIELD", "waiver_eligible","discount_eligible","GENDER", "Springer_agreement"]
    
    data_corr=combination[0].loc[:, (combination[0].columns != "FIELD") ].copy() 
    
    X = X[categorical_columns + numerical_columns]

    #hybrid dataset is imbalanced and we balance it with over sampleing method
    smote_nc = SMOTENC(categorical_features=[0,1,2,3,4], random_state=0)
    X, y = smote_nc.fit_resample(X, y)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    FOLD_NUM = 10
    folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=42)
    
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

    data_corr=combination[0].dropna(subset=['FIELD'])
    correlation_mat = data_corr.corr()
    
    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
        ]
    ) 

    rf = Pipeline(
        [
            ("preprocess", preprocessing),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    y_pred = cross_val_predict(rf, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)
    fold_iter = folds.split(X, y=y)
    perm_imp_list = []
    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):
        X_train, X_valid = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_valid = y[trn_idx], y[val_idx]         
        rf = Pipeline(
            [
                ("preprocess", preprocessing),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )
        rf.fit(X_train, y_train)
        
        result = permutation_importance(
            rf, X_valid, y_valid, n_repeats=10, random_state=42, n_jobs=2, scoring='accuracy'
        )
        perm_imp_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)
        perm_imp_list += [perm_imp_df]
    
    perm_importances_mean = pd.concat(perm_imp_list, axis=1)["importances_mean"]
    perm_importances_mean.columns = [f"fold_{i}" for i in range(FOLD_NUM)]
    perm_importances_mean["ave"] = perm_importances_mean.mean(axis=1)
    
    
    with open('cv_accuracy_importance_balanced_smote.txt', 'a') as f:
        sys.stdout = f
        print(combination[1])
        print(correlation_mat['oa_probility'])
        print('confusion matrix',conf_mat)
        print('classification report:',class_report)
        print(perm_importances_mean["ave"])
        print('\n')
        
    '''
    X.rename(columns={'FIELD': 'field','waiver_eligible': 'waiver eligible',"discount_eligible":"discount eligible","GENDER":"gender", 
        "Springer_agreement" :"OA agreement","COAUTHOR_COUNT_ITM":"number of co-authors","international_coauthor_percent":"proportion of international co-authors",
        "OA_AGIANST_CA_CITE":"ratio of citing OA against CA (current paper)","OA_against_CA_PUBLISH":"ratio of publishing OA against CA (prior papers)",
        "income_level":"income level of the country","JOURNAL_RANK2011_2016":"journal's ranking","APC_USD":"APC (us dollar)"},inplace=True)
    plt.rcParams.update({'font.size':60})
    plt.figure(figsize=(25,15))
    plt.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx],yerr=result.importances_std[sorted_idx])
    plt.xlabel("Permutation importances")
    plt.xlabel("Permutation importances")
    plt.title("(a)", loc='left')
    plt.savefig('feature_importance_'+combination[1]+'_balanced_smote.png', bbox_inches='tight', pad_inches=0.0)
    '''
   
if __name__ == '__main__':
    orig_stdout = sys.stdout
    dat = pd.read_csv("data.csv",sep=";")
    dat=dat.dropna(subset=["FIELD", "waiver_eligible","discount_eligible","GENDER", "Springer_agreement","age", "COAUTHOR_COUNT_ITM", "CNT_INTER_COLLAB","JOURNAL_RANK2011_2016", "OA_AGIANST_CA_CITE","OA_against_CA_PUBLISH","income_level","APC_USD"])
    FieldDummies = pd.get_dummies(dat['FIELD'], prefix='FIELD',dummy_na=False)
    #dat=dat.join(FieldDummies)
    dat.loc[:,'international_coauthor_percent']=dat['CNT_INTER_COLLAB']/dat['COAUTHOR_COUNT_ITM']
    dat.reset_index(drop=True, inplace=True)
    dat_hybrid=dat[(dat["OPEN_ACCESS"]=="Hybrid (Open Choice)")]
    dat.loc[:,'Hybrid']=np.where(dat["OPEN_ACCESS"]!="Hybrid (Open Choice)",0,1)
    dat_non_hybrid=dat[(dat["OPEN_ACCESS"]!="Hybrid (Open Choice)")]
    x_all=dat[["FIELD", "waiver_eligible","discount_eligible","GENDER", "Springer_agreement","age", "COAUTHOR_COUNT_ITM", "international_coauthor_percent","JOURNAL_RANK2011_2016", "OA_AGIANST_CA_CITE","OA_against_CA_PUBLISH","income_level","oa_probility"]]
    x_hybrid=dat_hybrid[["FIELD","GENDER", "Springer_agreement","age", "COAUTHOR_COUNT_ITM", "international_coauthor_percent","JOURNAL_RANK2011_2016", "OA_AGIANST_CA_CITE","OA_against_CA_PUBLISH","income_level","APC_USD","oa_probility"]]
    x_non_hybrid=dat_non_hybrid[["FIELD", "waiver_eligible","discount_eligible","GENDER", "Springer_agreement","age", "COAUTHOR_COUNT_ITM", "international_coauthor_percent","JOURNAL_RANK2011_2016", "OA_AGIANST_CA_CITE","OA_against_CA_PUBLISH","income_level","oa_probility"]]

    x_collection = {}
    x_collection=[(x_all,'x_all'), (x_hybrid,'x_hybrid'),(x_non_hybrid,'x_non_hybrid')]
    #x_collection=[(x_non_hybrid,'x_non_hybrid')]
   
    with Pool(3) as p:
        print(p.map(model, x_collection))
   



