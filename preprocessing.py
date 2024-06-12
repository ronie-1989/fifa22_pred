import wandb
import numpy as np
import pandas as pd
import xgboost as xgb
pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv', low_memory=False)

columns = list(df.columns)

f_target, f_home_team_name, f_away_team_name, f_match_date = [[feature] for feature in columns[1:5]]
f_league_name, f_league_id, f_is_cup, f_home_team_coach_id, f_away_team_coach_id = [[feature] for feature in columns[5:10]]

f_home_team_history_match_date = columns[10:20]
f_home_team_history_is_play_home = columns[20:30]
f_home_team_history_is_cup = columns[30:40]
f_home_team_history_goal = columns[40:50]
f_home_team_history_opponent_goal = columns[50:60]
f_home_team_history_rating = columns[60:70]
f_home_team_history_opponent_rating = columns[70:80]
f_home_team_history_coach = columns[80:90]
f_home_team_history_league_id = columns[90:100]
f_away_team_history_match_date = columns[100:110]
f_away_team_history_is_play_home = columns[110:120]
f_away_team_history_is_cup = columns[120:130]
f_away_team_history_goal = columns[130:140]
f_away_team_history_opponent_goal = columns[140:150]
f_away_team_history_rating = columns[150:160]
f_away_team_history_opponent_rating = columns[160:170]
f_away_team_history_coach = columns[170:180]
f_away_team_history_league_id = columns[180:190]

features_to_drop = f_match_date + f_home_team_coach_id + f_away_team_coach_id + f_home_team_history_match_date + f_home_team_history_coach + f_away_team_history_match_date + f_away_team_history_coach
features_boolean = f_is_cup + f_home_team_history_is_play_home + f_home_team_history_is_cup + f_away_team_history_is_play_home + f_away_team_history_is_cup
features_numerical = [f_home_team_history_goal] + [f_home_team_history_opponent_goal] + [f_home_team_history_rating] + [f_home_team_history_opponent_rating] + [f_away_team_history_goal] + [f_away_team_history_opponent_goal] + [f_away_team_history_rating] + [f_away_team_history_opponent_rating]
features_categorical = f_home_team_name + f_away_team_name + f_league_name + f_league_id + f_home_team_history_league_id + f_away_team_history_league_id
flat_features_numerical=[elem for sublist in features_numerical for elem in sublist]

def value_filler(df):
    for i in features_numerical:
        df[i]=df[i].apply(lambda x: x.fillna(df[i].mean(axis=1)))
    return df

#calculates the difference between match date and historical match 3 resp. 10
#in order to get a metric for short resp. longterm fatigue
#missing values get filled with median value
def date_converter(df):
    df['match_date']=pd.to_datetime(df['match_date'],infer_datetime_format=True)
    df['away_team_history_match_date_3']=pd.to_datetime(df['away_team_history_match_date_3'],infer_datetime_format=True)
    df['away_team_history_match_date_10']=pd.to_datetime(df['away_team_history_match_date_10'],infer_datetime_format=True)
    df['home_team_history_match_date_3']=pd.to_datetime(df['home_team_history_match_date_3'],infer_datetime_format=True)
    df['home_team_history_match_date_10']=pd.to_datetime(df['home_team_history_match_date_10'],infer_datetime_format=True)
    df['away_team_fatigue_short']=((df['match_date']-df['away_team_history_match_date_3']).dt.days).fillna(21)
    df['away_team_fatigue_long']=((df['match_date']-df['away_team_history_match_date_10']).dt.days).fillna(21)
    df['home_team_fatigue_short']=((df['match_date']-df['home_team_history_match_date_3']).dt.days).fillna(21)
    df['home_team_fatigue_long']=((df['match_date']-df['home_team_history_match_date_10']).dt.days).fillna(21)
    #drop no longer used columns
    df=df.drop(columns=['match_date']+f_home_team_history_match_date+f_away_team_history_match_date)
    return df

def is_cup_conversion(df):
    '''convert is_cup column from boolean to 0/1'''
    df.is_cup = df.is_cup.apply(lambda x: np.multiply(x, 1) )
    df['is_cup'].fillna(0, inplace=True) # missing value filled with 0, ie non-cup game
    return df

def has_coach_change(df):
    '''add features indicating whether home and away teams have changed coach'''
    #lambda function checks if coach has changed (and only returns true if the change isn't due to a missing value)
    df['home_has_coach_change'] = df.apply(lambda r: any([(r['home_team_coach_id']!=r[f'home_team_history_coach_{i}']) 
                                                    & (np.isnan(r[f'home_team_history_coach_{i}'])==False) for i in range(1,11) ]) , axis=1)
    #transforms booleans into 1/0
    df['home_has_coach_change'] = df['home_has_coach_change'].apply(lambda x: np.multiply(x, 1) )
    #just to be sure we fill values
    df['home_has_coach_change'].fillna(0, inplace=True)
    df['away_has_coach_change'] = df.apply(lambda r: any([(r['away_team_coach_id']!=r[f'away_team_history_coach_{i}']) 
                                                    & (np.isnan(r[f'away_team_history_coach_{i}'])==False) for i in range(1,11) ]) , axis=1)
    df['away_has_coach_change'] = df['away_has_coach_change'].apply(lambda x: np.multiply(x, 1) ).fillna(0, inplace=True)
    df['away_has_coach_change'].fillna(0, inplace=True)
    #drop no longer used columns
    df=df.drop(columns=f_home_team_coach_id + f_away_team_coach_id + f_home_team_history_coach  + f_away_team_history_coach)
    return df
#write function for imputing missing coach values
def fill_coach_id(df):
    df['home_team_coach_id']=df.apply(lambda row: row['home_team_history_coach_1'] if np.isnan(row['home_team_coach_id']) else row['home_team_coach_id'],
                                     axis=1)
    df['away_team_coach_id']=df.apply(lambda row: row['away_team_history_coach_1'] if np.isnan(row['away_team_coach_id']) else row['away_team_coach_id'],
                                     axis=1)
    df['home_team_coach_id'] = df['home_team_coach_id'].fillna(1)
    df['away_team_coach_id'] = df['away_team_coach_id'].fillna(1)
    return df

from sklearn.preprocessing import MinMaxScaler
def preprocessing(df):
    df=value_filler(df)
    df=fill_coach_id(df)
    df=has_coach_change(df)
    df=is_cup_conversion(df)
    df=date_converter(df)
    #perhaps include categorical features in the future
    df=df.drop(columns=features_categorical)
    df = df.dropna()
    df = df.drop_duplicates()
    df[flat_features_numerical]=df[flat_features_numerical].astype(float)
    df[features_boolean]=df[features_boolean].astype(float)
    #scaling
    scaler=MinMaxScaler()
    df[flat_features_numerical+list(df.columns[-4:])]=scaler.fit_transform(df[flat_features_numerical+list(df.columns[-4:])])
    return df

#data=preprocessing(df)  
#data.to_csv('preprocessed.csv',index=False)  