import numpy as np
import pandas as pd
import pickle
import time
import sys

######################################### Reading and Splitting the Data ###############################################

typeDict = {}
	
def handle_categorical_data(df, df_in,ksDatasetEncodedFile,inputEncodedFile):
    df['state'] = df['state'].map({'successful': 1,'failed':0})
    columns = df_in.columns.values

    for col in columns:
        conversion_vals = {}
	    
        def conversionToInteger(val):
            return conversion_vals[val]
        
        typeDict[col] = df[col].dtype	            
			
        if df[col].dtype != np.float64 and df[col].dtype != np.int64:
            colContents = df[col].values.tolist()
            uniqueElems = set(colContents)
            x = 0
            for elem in uniqueElems:
                if elem not in conversion_vals:
                    conversion_vals[elem] = x
                    x+=1
            df[col] = list(map(conversionToInteger, df[col]))
            df_in[col] = list(map(conversionToInteger, df_in[col]))
 
    df.to_csv(ksDatasetEncodedFile)
    df_in.to_csv(inputEncodedFile)


# Main function
def main():
    # read data
    fullInputFile = sys.argv[1]
    userInputFile = sys.argv[2]
    ksDatasetUnencodedFile = sys.argv[3]
    ksDatasetEncodedFile = sys.argv[4]
    inputUnencodedFile = sys.argv[5]
    inputEncodedFile = sys.argv[6]
    typesFile = sys.argv[7]

    dataframe = pd.read_csv(fullInputFile)
    dataframe = dataframe.drop(['current_currency','disable_communication','is_starrable','main_category'], axis=1)
    dataframe.head()
    dataframe.to_csv(ksDatasetUnencodedFile)

    dataframe_in = pd.read_csv(userInputFile)
    dataframe_in = dataframe_in.drop(['current_currency','disable_communication','is_starrable','main_category'], axis=1)
    dataframe_in.head()
    dataframe_in.to_csv(inputUnencodedFile)
    
    handle_categorical_data(dataframe, dataframe_in,ksDatasetEncodedFile,inputEncodedFile)
    pickle.dump(typeDict, open(typesFile, "wb" ))
    
if __name__ == "__main__":
    main()