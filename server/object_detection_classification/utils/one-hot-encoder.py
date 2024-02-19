import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder 

def create_dataframe(classes, df): 
    # Iterate through classes
    for class_name in classes:
        # Convert to category type if not already
        df[class_name] = df[class_name].astype('category')
        # Create a new column with numerical codes
        new_column_name = f'{class_name}_new'
        df[new_column_name] = df[class_name].cat.codes
        # Create an instance of OneHotEncoder 
        enc = OneHotEncoder()
        # One-hot encode the numerical codes
        enc_data = pd.DataFrame(enc.fit_transform(df[[new_column_name]]).toarray(), columns=enc.get_feature_names_out([new_column_name]))
        # Concatenate the one-hot encoded columns with the original DataFrame
        df = pd.concat([df, enc_data], axis=1)
        df = df.drop(columns=[class_name])
    print(df)
    return df
