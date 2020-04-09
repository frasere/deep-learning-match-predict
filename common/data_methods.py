import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_indiv_seasons(data_directory,cols):
    """
    Load csv match files from data directory
    
    Arg
    -----
    data_directory : str
    
    Out
    _____
    seasons : list of pandas df
    
    """
    
    match_result_csvs = glob.glob(data_directory + '\**\*.csv')
    match_result_lc = [pd.read_csv(i)[cols] for i in match_result_csvs]
    
    seasons = []
    for csv in match_result_lc:
        csv['Date'] = pd.to_datetime(csv['Date'],dayfirst=True)
        # drop matches without shot information and reset index
        csv = csv.dropna(subset=['HS','AS']).reset_index(drop=True)
        # one hot encode match result
        ftr_onehot = pd.get_dummies(csv['FTR'])
        csv = csv.join(ftr_onehot)
        # label encode match result (sanity check on predictions)
        le = LabelEncoder()
        csv['FTR_le'] = le.fit_transform(csv['FTR'])
        seasons.append(csv)
    
    return seasons


def train_test_split(match_data,x_cols,y_cols,scaler):
    # train test split and scale where necessary
    
    train_matches = match_data[match_data['Date']<'2019-07-01']
    test_matches = match_data[(match_data['Date']>='2019-07-01')]

    # inputs/ outputs 
    train_x = train_matches[x_cols].fillna(0)
    train_y = train_matches[y_cols]

    test_x = test_matches[x_cols].fillna(0)
    test_y = test_matches[['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','B365H','B365D','B365A']+y_cols]
    
    # scale features
    train_scaler = scaler
    test_scaler = scaler
    scaled_train_x = train_scaler.fit_transform(train_x)
    scaled_test_x = test_scaler.fit_transform(test_x)

    return scaled_train_x, train_y, scaled_test_x, test_y
    
