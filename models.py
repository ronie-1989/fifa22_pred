import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

class MLP(nn.Module):

    def __init__(self, dim_input):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, 3)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))
        return x

class BuildingBlock(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.fc3 = nn.Linear(dim_out, dim_out)
        self.shortcut = nn.Sequential()
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = x + self.shortcut(x)
        out = F.relu(out)
        return out

class Shortcut(nn.Module):

    def __init__(self, dim_input):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, 50)
        self.stack = nn.Sequential(*[BuildingBlock(50,30), BuildingBlock(30,10)])
        self.fc2 = nn.Linear(10, 3)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.stack(x)
        x = F.softmax(self.fc2(x))
        return x

class LSTM(nn.Module):

    def __init__(self, input_size, device = 'cpu', num_classes=3, hidden_size=10, num_layers=3):

        super().__init__()
        self.num_classes = num_classes # number of classes
        self.num_layers = num_layers # number of layers
        self.input_size = input_size # input dimension
        self.hidden_size = hidden_size # hidden dimension
        self.device = device # device

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) 
        self.fc1 = nn.Linear(hidden_size, 20)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
        self.batchnorm2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, num_classes)
        self.softmax = nn.Softmax()

    def forward(self,x):

        batch_size = x.size(0)

        # initialise hidden and internal states
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).requires_grad_().to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).requires_grad_().to(self.device)
        
        # propagate input through LSTM
        # .detach() is used to ensure stable BPTT
        output, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()))
        last_output = output[:,-1, :]

        # propagate through fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.batchnorm1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.batchnorm2(out)
        out = self.fc3(out)
        out = self.softmax(out)

        return out


class wc22_MLP(nn.Module):

    def __init__(self, dim_input):
        
        super().__init__()
        self.fc1 = nn.Linear(dim_input, 7)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(7, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):

        out = self.fc1(x)
        out = self.fc2(self.relu(out))
        out = self.softmax(out)

        return out


class wc22_model():

    def __init__(self, dim_input, lstm_file, xgb_file, wc22_file, data):
        
        super().__init__()
        self.lstm = LSTM(self.create_lstm_input(data).size(2))
        self.lstm.load_state_dict(torch.load(lstm_file, map_location=torch.device('cpu')))
        self.lstm.eval()
        self.xgb = xgb.Booster()
        self.xgb.load_model(xgb_file)
        self.mlp = wc22_MLP(6)
        self.mlp.load_state_dict(torch.load(wc22_file))
        self.mlp.eval()
    
    def forward(self, df):

        lstm_input = self.create_lstm_input(df)
        xgb_input = self.create_xgb_input(df)

        lstm_output = self.lstm.forward(lstm_input)
        xgb_output = self.xgb.predict(xgb_input, iteration_range=(0, self.xgb.best_iteration))

        mlp_input = torch.cat((lstm_output, torch.tensor(xgb_output)), -1)
        output = self.mlp.forward(mlp_input)

        return output

    def create_lstm_input(self, x_data):

        '''
        Input: pandas dataframe
        Output: input for lstm (torch tensor)
        '''

        # define sequential features
        historical_features = [

            'home_team_history_goal',
            'home_team_history_opponent_goal',
            'home_team_history_is_play_home', 
            'home_team_history_is_cup',
            'home_team_history_rating',
            'home_team_history_opponent_rating',

            'away_team_history_goal',
            'away_team_history_opponent_goal',
            'away_team_history_is_play_home', 
            'away_team_history_is_cup',
            'away_team_history_rating',
            'away_team_history_opponent_rating',
        ]

        # create 10 rows per game
        x_lstm_pivot = pd.wide_to_long(x_data, stubnames=historical_features, i=['id'], j='time', sep='_', suffix='\d+')

        # sort columns alphabetically
        cols = x_lstm_pivot.columns
        sorted_cols = cols.sort_values()
        x_lstm_pivot = x_lstm_pivot[sorted_cols]

        # sort the data
        x_lstm_pivot = x_lstm_pivot.sort_values(['id','time'])

        x_lstm = x_lstm_pivot.copy()

        # transform data to tensor
        x_lstm = torch.tensor(x_lstm.values).float()

        # resize data to 3-dimensional tensor
        x_lstm = x_lstm.view(-1, 10, x_lstm.shape[-1])

        return x_lstm

    def create_xgb_input(self, x_data):

        '''
        Input: pandas dataframe
        Output: input for xgboost
        '''

        # drop id column
        xgb_data = x_data.drop(['id'], axis=1)

        # load data into dmatrix
        x_xgb = xgb.DMatrix(xgb_data, enable_categorical=True)

        return x_xgb


def get_model(args, dim_input=0):

    '''
    - Returns requested model
    '''

    if args.model == 'mlp':
        return MLP(dim_input)
    elif args.model == 'shortcut':
        return Shortcut(dim_input)
    elif args.model == 'lstm':
        return LSTM(dim_input, args.device)
    elif args.model == 'wc22_mlp':
        return wc22_MLP(dim_input)
    elif args.model == 'wc22':
        return wc22_model(dim_input, args.lstm, args.xgb, args.wc22, args.data)
    else:
        raise Exception('invalid model')