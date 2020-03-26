import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_all_matches(data_directory):
    """
    Load csv match files from data directory
    
    Arg
    -----
    data_directory : str
    
    Out
    _____
    all_mathces : pandas df of all matches from all leagues
    
    """
    
    match_result_csvs = glob.glob(data_directory + '\**\*.csv')
    
    match_result_lc = [pd.read_csv(i) for i in match_result_csvs]

    # all matches in a dataframe
    all_matches = pd.concat(match_result_lc,sort=False)
    all_matches.Date = pd.to_datetime(all_matches.Date,dayfirst=True)
    
    # drop matches without shot information and reset index
    all_matches = all_matches.dropna(subset=['HS','AS']).reset_index(drop=True)
    
    # one hot encode match result
    ftr_onehot = pd.get_dummies(all_matches['FTR'])
    all_matches = all_matches.join(ftr_onehot)
    
    # label encode match result (sanity check on predictions)
    le = LabelEncoder()
    all_matches['FTR_le'] = le.fit_transform(all_matches['FTR'])
    
    return all_matches


def train_test_split(match_data,x_cols,y_cols,scaler):
    # train test split and scale where necessary
    
    train_matches = match_data[match_data['Date']<'2019-07-01']
    test_matches = match_data[(match_data['Date']>='2019-07-01')]

    # if using uncombined match data, need to combine the test set to make predictions
    if len(test_matches[test_matches[['Date','HomeTeam',
          'AwayTeam','FTR']].duplicated(keep=False)])>0:

        d = {key:'sum' for key in x_cols}
        test_matches = test_matches.groupby(list(match_data.columns)[:14],
                                            sort=False,
                                            as_index=False).agg(d)
    else:
        pass

    # inputs/ outputs 
    train_x = train_matches[x_cols].fillna(0)
    train_y = train_matches[y_cols]

    test_x = test_matches[x_cols].fillna(0)
    test_y = test_matches[['Date','HomeTeam','AwayTeam','FTHG','FTAG']+y_cols]
    
    # scale features
    train_scaler = scaler
    test_scaler = scaler
    scaled_train_x = train_scaler.fit_transform(train_x)
    scaled_test_x = test_scaler.fit_transform(test_x)

    return {'train_x':scaled_train_x,
            'train_y': train_y,
            'test_x': scaled_test_x,
            'test_y':test_y}
    
