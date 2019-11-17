import numpy as np
import pandas as pd
import pickle
import sys

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

#################KICKSTARTER SUCCESS/FAILURE ON USER DATA###################
def predict(encoded_df, rf_model_file):
    filename = rf_model_file
    rfc_model = pickle.load(open(filename, 'rb'))
    rfc_prediction = rfc_model.predict(encoded_df)
    return rfc_prediction

################VISUALIZATION #2 CALCULATIONS###################

def visualization2(unencoded_df,kmeans_model_file,backerClusterFile,goalClusterFile,ppbClusterFile,countryFile,lqFile,lmFile,catFile,slFile):
    filename = kmeans_model_file
    kmeans_model = pickle.load(open(filename, 'rb'))
    kmeans_user_df = unencoded_df.drop(['converted_pledged_amount','country','fx_rate','pledged','spotlight','staff_pick','duration','launched_quarter','launched_month','launched_week','launched_year','goal_reached','cat_name'], axis=1)
    cluster = kmeans_model.predict(kmeans_user_df)

    ##### BACKERS_COUNT statistic

    backerCluster = pickle.load(open(backerClusterFile, 'rb'))
    new_backers = backerCluster[cluster[0]]
    current_backers = unencoded_df.iloc[0]['backers_count']

    if int(new_backers) > int(current_backers):
	    backers_needed = int(new_backers)
    else:
	    backers_needed = int(current_backers)

    final_backers_count = backers_needed

    ##### GOAL statistic (continue to use cluster predicted in kmeans)

    goalCluster = pickle.load(open(goalClusterFile, 'rb'))
    new_goal = goalCluster[cluster[0]]
    current_goal = unencoded_df.iloc[0]['goal']

    if int(new_goal) < int(current_goal):
	    goal_needed = int(new_goal)
    else:
	    goal_needed = int(current_goal)
	
    final_goal = goal_needed

    ##### PLEDGE_PER_BACKER statistic (continue to use cluster predicted in kmeans)

    ppbCluster = pickle.load(open(ppbClusterFile, 'rb'))
    new_ppb = ppbCluster[cluster[0]]
    current_ppb = unencoded_df.iloc[0]['pledge_per_backer']	

    if int(new_ppb) > int(current_ppb):
	    ppb_needed = int(new_ppb)
    else:
	    ppb_needed = int(current_ppb)
	
    final_ppb = ppb_needed

    ##### COUNTRY, LAUNCHED_QUARTER, LAUNCHED_MONTH, CAT_NAME, SPOTLIGHT statistics

    country = pickle.load(open(countryFile, 'rb'))
    country_mode = country[0]

    lq = pickle.load(open(lqFile, 'rb'))
    lq_mode = lq[0]

    lm = pickle.load(open(lmFile, 'rb'))
    lm_mode = lm[0]

    cat = pickle.load(open(catFile, 'rb'))
    cat_mode = cat[0]

    sl = pickle.load(open(slFile, 'rb'))
    sl_mode = sl[0]
	
    #create Visualization #2 output CSV
    out_df = unencoded_df.drop(['converted_pledged_amount','fx_rate','pledged','staff_pick','duration','launched_week','launched_year','goal_reached'],axis=1)
    out_df_list = out_df.values.tolist()
    out_df_list.append([final_backers_count,country_mode,final_goal,sl_mode,lq_mode,lm_mode,final_ppb,cat_mode])
    out_df=pd.DataFrame(out_df_list,columns=['backers_count','country','goal','spotlight','launched_quarter','launched_month','pledge_per_backer','cat_name'])
    return out_df


################VISUALIZATION #3 CALCULATIONS###################
def visualization3(rf_model_file,visualization3File):
    featureList = ['backers_count','converted_pledged_amount','country','fx_rate','goal','pledged','spotlight','staff_pick','duration','launched_quarter','launched_month','launched_year','launched_week','goal_reached','pledge_per_backer','cat_name']
    rfc = pickle.load(open(rf_model_file, 'rb'))

    # Plot feature importance
    feature_importance = rfc.feature_importances_
    ind = np.argpartition(feature_importance, -16)
    
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    features = np.array(featureList)
    plt.yticks(pos, features[ind])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.savefig(visualization3File, format='png')

    return

######################## MAIN ####################################
def main():
    #User input CSV's
    inputUnencodedFile = sys.argv[1]
    inputEncodedFile = sys.argv[2]
    typesFile = sys.argv[3]

    #Model object files
    rf_model_file = sys.argv[4]
    kmeans_model_file = sys.argv[5]
    backerClusterFile = sys.argv[6]
    goalClusterFile = sys.argv[7]
    ppbClusterFile = sys.argv[8]	
    countryFile = sys.argv[9]	
    lqFile = sys.argv[10]	
    lmFile = sys.argv[11]	
    catFile = sys.argv[12]	
    slFile = sys.argv[13]	

    #Visualization and user error files
    predictionFile = sys.argv[14]	
    visualization2File = sys.argv[15]	
    visualization3File = sys.argv[16]
    errorFile = sys.argv[17]	

    user_df = pd.read_csv(inputUnencodedFile)
    user_df = user_df.drop(user_df.columns[0], axis=1)

    user_df_encoded = pd.read_csv(inputEncodedFile)
    user_df_encoded = user_df_encoded.drop(user_df_encoded.columns[0], axis=1)
	
	#Validate user input
    types = pickle.load(open(typesFile, 'rb'))
    message = "User Input Error for these columns: \n"
    accepted = ['goal', 'pledged', 'pledge_per_backer', 'fx_rate', 'converted_pledged_amount', 'goal_reached']

    for column in user_df.columns.values:
        if user_df[column].dtype != types[column]:
            if column in accepted and user_df[column].dtype == np.int64:
                continue
            else:
                message = message + str(column) + " expected " + str(types[column]) + "\n"

    if message == "User Input Error for these columns: \n": 
        message = "Valid User Input"
        with open(errorFile, "w") as file_object:
            file_object.write(message)
    else:
        with open("inputFeedback.txt", "w") as file_object:
            file_object.write(message)	
        return
		
    #Obtain and Output Prediction of Success
    state_prediction = predict(user_df_encoded, rf_model_file)

    prediction = []
    if state_prediction[0] == 1:
        prediction.append('successful')
    else:
        prediction.append('failed')

    prediction_out = pd.DataFrame(prediction,columns=['prediction'])
    prediction_out.to_csv(predictionFile)

    #Create Visualization 2
    out_df = visualization2(user_df, kmeans_model_file,backerClusterFile,goalClusterFile,ppbClusterFile,countryFile,lqFile,lmFile,catFile,slFile)
    out_df.drop(out_df.columns[0], axis=1)
    out_df.to_csv(visualization2File)

    #Create Visualization 3
    visualization3(rf_model_file,visualization3File)

    return

if __name__ == "__main__":
    main()
