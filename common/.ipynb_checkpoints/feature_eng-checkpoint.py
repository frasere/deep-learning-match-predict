import glob
import pandas as pd



def cumsum_reset(v):
    # cumsum a series whilst resetting to zero
    
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    result = v.where(v.notnull(), reset).cumsum()
    
    return result



def feature_builder(team,matches,window):
    # create features for each team
    
    # select team matches
    team_df = matches[((matches['HomeTeam']==team)|(matches['AwayTeam']==team))]
    
    # sort chronologically
    team_df = team_df.sort_values(['Date'])
    
    # get team name from team_df
    team = (
        team_df.groupby(['HomeTeam'])['HomeTeam']
            .count().sort_values(ascending=False)
            .index[0]
           )
    
    # home matches df
    home_df = team_df.loc[team_df['HomeTeam']==team]
    
    # home unbeaten
    home_df.loc[(home_df['FTR']== 'H')|(home_df['FTR']=='D'),
               'HU_counter'] = 1
    
    home_df['HU'] = cumsum_reset(home_df['HU_counter']).shift().fillna(0)
    
    # home goals scored and conceded rolling mean
    home_df['HG_rm'] = home_df['FTHG'].rolling(window).mean().shift()
    home_df['HC_rm'] = home_df['FTAG'].rolling(window).mean().shift()
    # GD rolling mean
    home_df['HGD_rm'] = home_df['HG_rm']-home_df['HC_rm']
    
    # shot target and conversion %
    home_df['HST%'] = round(home_df['HST']/home_df['HS'],2)  # home shot target %
    home_df['HSTC%'] = round(home_df['AST']/home_df['AS'],2)  # home shot target conc %
    home_df['HGS%'] = round(home_df['FTHG']/home_df['HS'],2) # home goals per shot
    home_df['HGSC%'] = round(home_df['FTAG']/home_df['AS'],2) # home goals per shot conc
    
    # shot rolling means
    home_df['HST%_rm'] = home_df['HST%'].rolling(window).mean().shift()
    home_df['HSTC%_rm'] = home_df['HSTC%'].rolling(window).mean().shift()
    home_df['HGS%_rm'] = home_df['HGS%'].rolling(window).mean().shift()
    home_df['HGSC%_rm'] = home_df['HGSC%'].rolling(window).mean().shift()
    
    
    ##########################################################
    
    # away matches df
    away_df = team_df.loc[team_df['AwayTeam']==team]
    
    # away unbeaten
    away_df.loc[(away_df['FTR']=='A')|(away_df['FTR']=='D'),
               'AU_counter'] = 1
    
    away_df['AU'] = cumsum_reset(away_df['AU_counter']).shift().fillna(0)
    
    # away goals scored and conceded rolling mean
    away_df['AG_rm'] = away_df['FTAG'].rolling(5).mean().shift()
    away_df['AC_rm'] = away_df['FTHG'].rolling(5).mean().shift()
    # GD rolling mean
    away_df['AGD_rm'] = away_df['AG_rm']-away_df['AC_rm']
        
    # shot target and conversion %
    away_df['AST%'] = round(away_df['AST']/away_df['AS'],2)  # home shot target %
    away_df['ASTC%'] = round(away_df['HST']/away_df['HS'],2)  # home shot target conc %
    away_df['AGS%'] = round(away_df['FTAG']/away_df['AS'],2) # home goals per shot
    away_df['AGSC%'] = round(away_df['FTHG']/away_df['HS'],2) # home goals per shot conc
    
    # shot rolling means
    away_df['AST%_rm'] = away_df['AST%'].rolling(window).mean().shift()
    away_df['ASTC%_rm'] = away_df['ASTC%'].rolling(window).mean().shift()
    away_df['AGS%_rm'] = away_df['AGS%'].rolling(window).mean().shift()
    away_df['AGSC%_rm'] = away_df['AGSC%'].rolling(window).mean().shift()
    
    ###############################################################
    
    # merge home and away dataframes
    merged_df = home_df.merge(away_df,how='outer')
    
        
    return merged_df


def combine_matches(df,fixed_feats):
    # function to combine home and away matches in feature builder
    
    summing_feats = [x for x in df.columns if x not in fixed_feats]
    d = {key:'sum' for key in summing_feats}
    comb_df = (df.groupby(fixed_feats,sort=False, as_index=False).agg(d))
    
    return comb_df