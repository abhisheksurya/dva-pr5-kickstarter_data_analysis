import numpy as np
import pandas as pd
import time
import pickle
import sys

from sklearn import model_selection
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    ksDatasetUnencodedFile = sys.argv[1]
    ksDatasetEncodedFile = sys.argv[2]
    rf_model_file = sys.argv[3]
    kmeans_model_file = sys.argv[4]
    backerClusterFile = sys.argv[5]
    goalClusterFile = sys.argv[6]
    ppbClusterFile = sys.argv[7]	
    countryFile = sys.argv[8]	
    lqFile = sys.argv[9]	
    lmFile = sys.argv[10]	
    catFile = sys.argv[11]	
    slFile = sys.argv[12]	
######################################### Reading and Splitting the Data ###############################################
    # read data
    df = pd.read_csv(ksDatasetEncodedFile)
    df = df.drop(df.columns[0], axis=1)
    
    # Separate out the x_data and y_data.
    x_data = df.loc[:, df.columns != "state"]
    y_data = df.loc[:, "state"]
    
    random_state = 100
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, train_size=0.6,shuffle=True, random_state=100)

################LOGISTIC REGRESSION################
    # Declare a logistic regression classifier.
    # Parameter regularization coefficient C described above.
    lr = LogisticRegression(penalty='l2', solver='liblinear')
    #lr = LogisticRegression()
    
    # Fit the model.
    fit = lr.fit(x_train, y_train)
    lr_train = lr.predict(x_train)
    lr_test = lr.predict(x_test)
    
    #print('Logistic Regression Cross Matrix:', pd.crosstab(lr_test, y_test))
    #print('LR score:', lr.score(x_test, y_test))
    #print('LR train accuracy:', accuracy_score(y_train,lr_train.round()))
    #print('LR test accuracy:', accuracy_score(y_test,lr_test.round()))


################RANDOM FOREST################
    rfc = RandomForestClassifier(n_jobs=-1, n_estimators=20)
    #rfc = RandomForestClassifier()
    
    rfc.fit(x_train, y_train)
    rfc_train = rfc.predict(x_train)
    rfc_test = rfc.predict(x_test)
    
    filename = rf_model_file
    pickle.dump(rfc, open(filename, 'wb'))
    
    #print('RFC score:', rfc.score(x_test, y_test))
    #print('RFC train accuracy:', accuracy_score(y_train,rfc_train.round()))
    #print('RFC test accuracy:', accuracy_score(y_test,rfc_test.round()))
    
################KMEANS CLUSTERING##############
    kmeansData = x_train.drop(['converted_pledged_amount','country','fx_rate','pledged','spotlight','staff_pick','duration','launched_quarter','launched_month','launched_week','launched_year','goal_reached','cat_name'], axis=1)
    
    kmeans = KMeans(n_clusters=50, random_state=0).fit(kmeansData)
    kmeansData['cluster_label'] = kmeans.labels_
    kmeansData['success_rate'] = y_train
    
    filename = kmeans_model_file
    pickle.dump(kmeans, open(filename, 'wb'))
    
    backers_Cluster = {}
    goal_Cluster = {}
    ppb_Cluster = {}
    success_only = kmeansData.loc[kmeansData['success_rate'] == 1]
    
    for cluster in kmeans.labels_:
        cluster_only = success_only.loc[success_only['cluster_label'] == cluster]
        if cluster_only.empty:
            cluster_only = kmeansData.loc[kmeansData['cluster_label'] == cluster]		
        backers_Cluster[cluster] = cluster_only['backers_count'].mean()
        goal_Cluster[cluster] = cluster_only['goal'].mean()
        ppb_Cluster[cluster] = cluster_only['pledge_per_backer'].mean()

    pickle.dump(backers_Cluster, open(backerClusterFile, "wb" ))
    pickle.dump(goal_Cluster, open(goalClusterFile, "wb" ))
    pickle.dump(ppb_Cluster, open(ppbClusterFile, "wb" ))
    
################FILTERING##################
    
    dfU_all = pd.read_csv(ksDatasetUnencodedFile)
    dfU = dfU_all.loc[dfU_all['state'] == "successful"]
    
    country_mode = dfU.loc[:,"country"].mode()
    #success = dfU.loc[dfU['country'] == country_mode]
    #all = dfU_all.loc[dfU_all['country'] == country_mode]
    pickle.dump(country_mode, open(countryFile, "wb" ))
    #print(country_mode[0])
    
    lq_mode = dfU.loc[:,"launched_quarter"].mode()
    pickle.dump(lq_mode, open(lqFile, "wb" ))
    #print(lq_mode[0])
    
    lm_mode = dfU.loc[:,"launched_month"].mode()
    pickle.dump(lm_mode, open(lmFile, "wb" ))
    #print(lm_mode[0])
    
    cat_mode = dfU.loc[:,"cat_name"].mode()
    pickle.dump(cat_mode, open(catFile, "wb" ))
    #print(cat_mode[0])
    
    sl_mode = dfU.loc[:,"spotlight"].mode()
    pickle.dump(sl_mode, open(slFile, "wb" ))
    #print(sl_mode[0])
    
    	
''' Not using due to overfitting training data
################GRADIENT BOOSTING MACHINE################
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=2, loss='deviance',subsample=1.0)
    #clf = GradientBoostingClassifier()
    
    clf.fit(x_train,y_train)
    
    #print('\n Percentage accuracy for Gradient Boosting Classifier')
    predict_train = clf.predict(x_train)
    predict_test = clf.predict(x_test)
    
    table_train = pd.crosstab(y_train, predict_train, margins=True)
    table_test = pd.crosstab(y_test, predict_test, margins=True)
    
    train_tI_errors = table_train.loc[0.0,1.0] / table_train.loc['All','All']
    train_tII_errors = table_train.loc[1.0,0.0] / table_train.loc['All','All']
    
    test_tI_errors = table_test.loc[0.0,1.0]/table_test.loc['All','All']
    test_tII_errors = table_test.loc[1.0,0.0]/table_test.loc['All','All']
        
    train_accuracy = 1 - (train_tI_errors + train_tII_errors)
    test_accuracy = 1 - (test_tI_errors + test_tII_errors)
        
    #print((
        'Training set accuracy:\n'
        'Overall Accuracy: {}\n'
        'Test set accuracy:\n'
        'Overall Accuracy: {}\n'
        ).format(train_accuracy, test_accuracy))
    
    
    ind = np.argpartition(feature_importance, -3)[-3:]
    #print(feature_importance[ind])
    topFeatures = [0.14380596, 0.43407818, 0.26980206]
    #featureList = ['backers_count','goal_reached','spotlight']
    features = np.array(featureList)
    #print(features[ind])
'''

if __name__ == "__main__":
    main()
