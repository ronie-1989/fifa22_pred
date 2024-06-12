import numpy as np
import pandas as pd
import importlib
import torch
import time
import argparse

from preprocessing import preprocessing
from models import get_model 
from wc22_data import is_cup, league_ids, ratings
import wc22_games
import wc22_teams

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-lstm', default='', type=str)
parser.add_argument('-xgb', default='', type=str)
parser.add_argument('-wc22', default='', type=str)
args = parser.parse_args()
args.model = 'wc22'

# read train data to get feature names
df = pd.read_csv('train.csv', low_memory=False)
columns = list(df.columns)

# store teams and game states
teams = {}
games = []

# store predictions
First = {}
Second = {}
Third = {}
Fourth = {}

# scale team ratings
min_rating = min(ratings.values())
max_rating = max(ratings.values())
ratings = { key : (value - min_rating) / (max_rating - min_rating) for key, value in ratings.items()}


class Team:

    '''
    - stores a team's state during the wc
    - tracks points in group stage and handles game history
    '''

    def __init__(self, name, coach_id, history):
        self.name = name
        self.coach_id = coach_id
        self.history = history
        self.points = 0


class HistoryGame:

    '''
    - models one of the ten previous games per team
    - the results are deterministic -> no prediction
    '''

    def __init__(self, game):
        self.match_date = game['match_date']
        self.is_play_home = game['is_play_home']
        self.is_cup = game['is_cup']
        self.goal = game['goal']
        self.opponent_goal = game['opponent_goal']
        self.rating = game['rating'] 
        self.opponent_rating = game['opponent_rating']
        self.coach = game['coach']
        self.league_id = game['league_id']


class WorldCupGame:

    '''
    - models a game in the tournament
    - the results are stochastic -> prediction needed
    - a predicted wc game leads to the creation of two history games (one per team)
    '''

    def __init__(self, game, target=None, league_name='World Cup', league_id=0, is_cup=True):
        self.id = game['id']
        self.home_team = None if game['home_team'] == None else teams[game['home_team']]
        self.away_team = None if game['away_team'] == None else teams[game['away_team']]
        self.match_date = game['match_date']
        self.target = target
        self.league_name = league_name
        self.league_id = league_id
        self.is_cup = is_cup


def reset_wc22():

    '''
    - resets team and game states
    - needs to be called before each wc modelling
    '''

    global teams, games

    teams = {}
    games = []

    for name in wc22_teams.names:
        history = importlib.import_module(f'wc22_history_{name}')
        teams[name] = Team(name=name, coach_id=0, history=[HistoryGame(game) for game in history.games])
        
    for game in wc22_games.data:
        games.append(WorldCupGame(game))


def create_model_inputs(games):

    '''
    input: list of wc games
    output: pandas dataframe in form of the original train.csv dataset
    '''

    hist_features = [
        'match_date',
        'is_play_home',
        'is_cup',
        'goal',
        'opponent_goal',
        'rating',
        'opponent_rating',
        'coach',
        'league_id'
    ]

    data = {}

    for feature in columns:
        data[feature] = []

    for game in games:

        data['id'].append(game.id)
        data['target'].append(game.target)
        data['home_team_name'].append(game.home_team.name)
        data['away_team_name'].append(game.away_team.name)
        data['match_date'].append(game.match_date)
        data['league_name'].append(game.league_name)
        data['league_id'].append(game.league_id)
        data['is_cup'].append(game.is_cup)
        data['home_team_coach_id'].append(game.home_team.coach_id)
        data['away_team_coach_id'].append(game.away_team.coach_id)

        for feature in hist_features:
                for i in range(1,11):
                    data[f'home_team_history_{feature}_{i}'].append(getattr(game.home_team.history[i-1], feature))
                    data[f'away_team_history_{feature}_{i}'].append(getattr(game.away_team.history[i-1], feature))
    
    # create dataframe from given game data
    data_set = pd.DataFrame(data, columns=columns)

    # drop the target (it is none yet)
    data_set = data_set.drop('target', axis=1)

    # apply the preprocessing
    data_set = preprocessing(data_set)

    return data_set


def predict(games):

    '''
    input: list of wc games
    - predicts each game based on the final wc22 model
    - updates the team and game states
    '''
    
    # prepare model inputs
    inputs = create_model_inputs(games)
    
    for idx in inputs.index:

        game = games[idx]

        # predict game with wc22 model
        outputs = model.forward(inputs.iloc[idx:idx+1])
        probs = outputs[0].cpu().detach().numpy()

        # randomly draw the result based on predicted probabilities     
        result = np.random.choice(['home','draw','away'], p=probs)

        # save result
        game.target = result
        
        # save goals (we always assume 2:1 / 1:1 / 1:2)
        home_team_goals = 2 if result == 'home' else 1
        away_team_goals = 2 if result == 'away' else 1

        # add the predicted result to each team's history
        home_team_history_game = {
            'match_date': game.match_date,
            'is_play_home': True,
            'is_cup': True,
            'goal': home_team_goals,
            'opponent_goal': away_team_goals,
            'rating': ratings[game.home_team.name],
            'opponent_rating': ratings[game.away_team.name],
            'coach': 0,
            'league_id': league_ids['world_cup']
        }
        
        away_team_history_game = {
            'match_date': game.match_date,
            'is_play_home': False,
            'is_cup': True,
            'goal': away_team_goals,
            'opponent_goal': home_team_goals,
            'rating': ratings[game.away_team.name],
            'opponent_rating': ratings[game.home_team.name],
            'coach': 0,
            'league_id': league_ids['world_cup']
        }
        
        game.home_team.history.pop()
        game.away_team.history.pop()
        
        game.home_team.history.insert(0, HistoryGame(home_team_history_game))
        game.away_team.history.insert(0, HistoryGame(away_team_history_game))


def predict_gameday_1():
    predict(games[1:17])
    

def predict_gameday_2():
    predict(games[17:33])
    

def predict_gameday_3():
    predict(games[33:49])
    

def predict_round_of_16():
    predict(games[49:57])
    

def predict_quaterfinals():
    predict(games[57:61])
    

def predict_semifinals():
    predict(games[61:63])
    

def predict_finals():
    predict(games[63:65])


def make_round_of_16():

    '''
    - determines who faces whom in the round of 16 based on tournament bracket
    - tournament bracket: 'https://www.worldcupbrackets.info'
    '''
    
    # determine points of each team
    for game in games[1:49]:
        
        if game.target == 'home':
            game.home_team.points += 3
            
        elif game.target == 'away':
            game.away_team.points += 3
            
        elif game.target == 'draw':
            game.home_team.points += 1
            game.away_team.points += 1
            
        else:
            raise Exception('invalid game result')
    
    # create groups
    A = [teams[wc22_teams.names[i]] for i in range(0,4)]
    B = [teams[wc22_teams.names[i]] for i in range(4,8)]
    C = [teams[wc22_teams.names[i]] for i in range(8,12)]
    D = [teams[wc22_teams.names[i]] for i in range(12,16)]
    E = [teams[wc22_teams.names[i]] for i in range(16,20)]
    F = [teams[wc22_teams.names[i]] for i in range(20,24)]
    G = [teams[wc22_teams.names[i]] for i in range(24,28)]
    H = [teams[wc22_teams.names[i]] for i in range(28,32)]
    
    # sort groups by points
    for group in [A,B,C,D,E,F,G,H]:
        group.sort(key = lambda x: (x.points, x.history[0].rating), reverse=True)
        
    # make round of 16
    games[49].home_team, games[49].away_team = A[0], B[1]
    games[50].home_team, games[50].away_team = C[0], D[1]
    games[51].home_team, games[51].away_team = B[0], A[1]
    games[52].home_team, games[52].away_team = D[0], C[1]
    games[53].home_team, games[53].away_team = E[0], F[1]
    games[54].home_team, games[54].away_team = G[0], H[1]
    games[55].home_team, games[55].away_team = F[0], E[1]
    games[56].home_team, games[56].away_team = H[0], G[1]


def make_quaterfinals():

    '''
    - determines who faces whom in the quaterfinals based on tournament bracket
    - tournament bracket: 'https://www.worldcupbrackets.info'
    '''

    games[57].home_team, games[57].away_team = winner_of(games[49]), winner_of(games[50])
    games[58].home_team, games[58].away_team = winner_of(games[53]), winner_of(games[54])
    games[59].home_team, games[59].away_team = winner_of(games[51]), winner_of(games[52])
    games[60].home_team, games[60].away_team = winner_of(games[55]), winner_of(games[56])


def make_semifinals():

    '''
    - determines who faces whom in the semifinals based on tournament bracket
    - tournament bracket: 'https://www.worldcupbrackets.info'
    '''

    games[61].home_team, games[61].away_team = winner_of(games[57]), winner_of(games[58])
    games[62].home_team, games[62].away_team = winner_of(games[59]), winner_of(games[60])


def make_finals():

    '''
    - determines who faces whom in the finals based on tournament bracket
    - tournament bracket: 'https://www.worldcupbrackets.info'
    '''

    games[63].home_team, games[63].away_team = loser_of(games[61]), loser_of(games[62])
    games[64].home_team, games[64].away_team = winner_of(games[61]), winner_of(games[62])


def winner_of(game):

    '''
    - input: wc game
    - output: predicted winner
    '''
    
    if game.target == 'home':
        return game.home_team
    
    elif game.target == 'away':
        return game.away_team
    
    elif game.target == 'draw':

        # after the group stage there has to be a winner
        # if the model predicts a draw we chose the better rated team as the winner
        if game.home_team.history[0].rating > game.away_team.history[0].rating:
            return game.home_team
        else:
            return game.away_team
        
    else:
        raise Exception('invalid game result')


def loser_of(game):

    '''
    - input: wc game
    - output: predicted loser
    '''
    
    if game.target == 'home':
        return game.away_team
    
    elif game.target == 'away':
        return game.home_team
    
    elif game.target == 'draw':

        # after the group stage there has to be a loser
        # if the model predicts a draw we chose the worse rated team as the loser
        if game.home_team.history[0].rating > game.away_team.history[0].rating:
            return game.away_team
        else:
            return game.home_team
        
    else:
        raise Exception('invalid game result')


def predict_world_cup():

    '''
    - models the worldcup
    - returns best 4 teams
    '''
    
    reset_wc22()

    predict_gameday_1()
    predict_gameday_2()
    predict_gameday_3()
    
    make_round_of_16()
    predict_round_of_16()
    
    make_quaterfinals()
    predict_quaterfinals()
    
    make_semifinals()
    predict_semifinals()
    
    make_finals()
    predict_finals()

    return winner_of(games[64]).name, loser_of(games[64]).name, winner_of(games[63]).name, loser_of(games[63]).name

# configure model
reset_wc22()
args.data = create_model_inputs(games[1:5])
model = get_model(args)

# configure wc modelling
for team in teams:
    First[team] = 0
    Second[team] = 0
    Third[team] = 0
    Fourth[team] = 0

# model wc 1000 times
for i in range(1000):
    first, second, third, fourth = predict_world_cup()
    print(str(i) + ': winner: ' + winner_of(games[64]).name)
    First[first] += 1
    Second[second] += 1
    Third[third] += 1
    Fourth[fourth] += 1

First = dict(sorted(First.items(), key=lambda item: item[1]))
Second = dict(sorted(Second.items(), key=lambda item: item[1]))
Third = dict(sorted(Third.items(), key=lambda item: item[1]))
Fourth = dict(sorted(Fourth.items(), key=lambda item: item[1]))

# print results
print('First: ', First)
print('Second: ', Second)
print('Third: ', Third)
print('Fourth: ', Fourth)