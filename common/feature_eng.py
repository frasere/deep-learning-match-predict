import glob
import pandas as pd



def cumsum_reset(v):
    # cumsum a series whilst resetting to zero
    
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    result = v.where(v.notnull(), reset).cumsum()
    
    return result


def home_feature_build(df,window):
    
    # home goals scored and conceded rolling mean
    df['HG_rm'] = df['FTHG'].rolling(window,min_periods=1).mean().shift()
    df['HC_rm'] = df['FTAG'].rolling(window,min_periods=1).mean().shift()
    # GD rolling mean
    df['HGD_rm'] = df['HG_rm']- df['HC_rm']
    # shot rolling means
    df['HST%_rm'] = round(df['HST']/df['HS'],2).rolling(window,min_periods=1).mean().shift()
    df['HSTC%_rm'] = round(df['AST']/df['AS'],2).rolling(window,min_periods=1).mean().shift()
    df['HGS%_rm'] = round(df['FTHG']/df['HS'],2).rolling(window,min_periods=1).mean().shift()
    df['HGSC%_rm'] = round(df['FTAG']/df['AS'],2).rolling(window,min_periods=1).mean().shift()
    # points
    df['HP'] = 0
    df.loc[(df['FTR']=='H'),'HP'] = 3
    df.loc[(df['FTR']=='D'),'HP'] = 1
    df['HP'] = df['HP'].cumsum().shift().fillna(0)
    
    return df


def away_feature_build(df,window):
    # away goals scored and conceded rolling mean
    df['AG_rm'] = df['FTAG'].rolling(window,min_periods=1).mean().shift()
    df['AC_rm'] = df['FTHG'].rolling(window,min_periods=1).mean().shift()
    # GD rolling mean
    df['AGD_rm'] = df['AG_rm']-df['AC_rm']
    # shot rolling means
    df['AST%_rm'] = round(df['AST']/df['AS'],2).rolling(window,min_periods=1).mean().shift()
    df['ASTC%_rm'] = round(df['HST']/df['HS'],2).rolling(window,min_periods=1).mean().shift()
    df['AGS%_rm'] = round(df['FTAG']/df['AS'],2).rolling(window,min_periods=1).mean().shift()
    df['AGSC%_rm'] = round(df['FTHG']/df['HS'],2).rolling(window,min_periods=1).mean().shift()
    # points
    df['AP'] = 0
    df.loc[(df['FTR']=='A'),'AP'] = 3
    df.loc[(df['FTR']=='D'),'AP'] = 1
    df['AP'] = df['AP'].cumsum().shift().fillna(0)
    
    return df


def master_feature_builder(seasons,window):
    # seperate each season into individual team home and away dfs
    list_teams = []
    list_home_dfs = []
    list_away_dfs = []
    for i in range(len(seasons)):
        list_teams.append(seasons[i]['HomeTeam'].unique())
        for j in list_teams[i]:
            list_home_dfs.append(seasons[i][(seasons[i]['HomeTeam']==j)])
            list_away_dfs.append(seasons[i][(seasons[i]['AwayTeam']==j)])
    
    # calculate home and away features
    home_list = [home_feature_build(df,window) for df in list_home_dfs]
    away_list = [away_feature_build(df,window) for df in list_away_dfs]
    all_homes = pd.concat(home_list)
    all_aways = pd.concat(away_list)
    all_matches = all_homes.append(all_aways).reset_index(drop=True)
    
    # combine home and away into dataframe
    fixed_feats = ['Div','Date', 'HomeTeam', 'AwayTeam','FTR', 'FTHG', 'FTAG', 'HS','AS', 'HST', 'AST', 'A', 'D', 'H',
                   'FTR_le','B365H','B365A','B365D']
    summing_feats = ['HG_rm','HC_rm','HGD_rm','HST%_rm','HSTC%_rm','HGS%_rm','HGSC%_rm','AG_rm',
                     'AC_rm','AGD_rm','AST%_rm','ASTC%_rm','AGS%_rm','AGSC%_rm','HP','AP']
    d = {key:'sum' for key in summing_feats}
    comb_df = (all_matches.groupby(fixed_feats,sort=False, as_index=False).agg(d))
    
    return comb_df