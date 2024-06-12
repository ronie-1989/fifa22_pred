import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import wandb
import random
import argparse
import numpy as np
import pandas as pd

from os.path import exists
from models import get_model 
from fastprogress import master_bar, progress_bar
from sklearn.model_selection import train_test_split

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-bs', default=128, type=int, help='batch size')                    # batch size
parser.add_argument('-dataset', default='', type=str, help='file path to the dataset')  # dataset
parser.add_argument('-epochs', default=50, type=int, help='number of epochs')           # epochs
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')               # learning rate
parser.add_argument('-model', default='mlp', type=str, help='choose mlp/shortcut/lstm') # model
parser.add_argument('-mom', default=0.9, type=float, help='momentum for sgd')           # momentum
parser.add_argument('-optimizer', default='sgd', type=str, help='choose sgd/adam')      # optimizer
parser.add_argument('-rs', default=22, type=int, help='random seed')                    # random seed
parser.add_argument('-wd', default=0, type=float, help='weight decay')                  # weight decay

args = parser.parse_args()
random.seed(args.rs)

# check arguments
if args.bs <= 0:
    raise Exception('invalid batch size')
elif not exists(args.dataset):
    raise Exception('invalid dataset')
elif args.epochs <= 0:
    raise Exception('invalid epochs')
elif args.lr <= 0:
    raise Exception('invalid learning rate')
elif args.mom < 0 or args.mom > 1:
    raise Exception('invalid momentum')
elif args.wd < 0:
    raise Exception('invalid weight decay')

# login to weights and biases
wandb.login(key='147686d07ab47cb770a0957694c8a6f896671f2c')

# check GPU access
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')
if device == 'cuda':
    cudnn.benchmark = True
else:
    print('Are you sure you want to run on CPU?')

# save the device to args
args.device = device

# read dataset
data = pd.read_csv(args.dataset, low_memory=False)

if args.model == 'lstm':

    #############################
    # data preparation for lstm #
    #############################

    # define targets and features
    targets = data.columns[1:2]
    features = np.setdiff1d(data.columns, targets)

    # split data in train and test set
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[targets], test_size=0.2, shuffle= False)

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
    x_train_pivot = pd.wide_to_long(x_train, stubnames=historical_features, i=['id'], j='time', sep='_', suffix='\d+')
    x_test_pivot = pd.wide_to_long(x_test, stubnames=historical_features, i=['id'], j='time', sep='_', suffix='\d+')

    # sort columns alphabetically
    cols = x_train_pivot.columns
    sorted_cols = cols.sort_values()
    x_train_pivot = x_train_pivot[sorted_cols]
    x_test_pivot = x_test_pivot[sorted_cols]

    # sort the data first by id, then by time
    x_train_pivot = x_train_pivot.sort_values(['id','time'])
    x_test_pivot = x_test_pivot.sort_values(['id','time'])

    x_train = x_train_pivot.copy()
    x_test = x_test_pivot.copy()

    # one hot encode target
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    # transform data to tensor
    x_train = torch.tensor(x_train.values).float()
    y_train = torch.tensor(y_train.values).float()
    x_test = torch.tensor(x_test.values).float()
    y_test = torch.tensor(y_test.values).float()

    # resize data to 3-dimensional tensor
    x_train = x_train.view(-1, 10, x_train.shape[-1])
    x_test = x_test.view(-1, 10, x_test.shape[-1])

else:

    #######################################
    # data preparation for mlp / shortcut #
    #######################################

    # one hot encode target
    data_set = data[['id']]
    data_set = data_set.join(pd.get_dummies(data['target']))
    data_set = data_set.join(data[data.columns[2:]])

    print('data: ', data_set.shape)

    # split data in targets and features
    targets = data_set.columns[1:4]
    features = data_set.columns[4:]

    # transform data to tensor
    x = torch.tensor(data_set[features].values).float()
    y = torch.tensor(data_set[targets].values).float()

    # split data in train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=args.rs)

# define dataset
class DataSet():

    def __init__(self,x,y):
        self.x_train = x
        self.y_train = y

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self,idx):
        return self.x_train[idx], self.y_train[idx]

# configure dataloaders
train_loader = torch.utils.data.DataLoader(DataSet(x_train,y_train), batch_size=args.bs, shuffle=True)
test_loader =  torch.utils.data.DataLoader(DataSet(x_test,y_test), batch_size=args.bs, shuffle=True)

# configure model
if args.model == 'lstm':
    model = get_model(args, x_train.size(2))
else:
    model = get_model(args, len(features))

model.float()
model.to(device)

# configure criterion
criterion = nn.CrossEntropyLoss()

# configure optimizer
if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd) 
else:
    raise Exception('invalid optimizer')


def test_model(model):

    '''
    - Handles model training mode
    - Computes metrics: accuracy, per class accuracy, confusion matrix
    '''
    
    train_mode = model.training
    
    if train_mode:
        model.eval()
    
    # set up confusion matrix
    confusion = np.zeros((3,3), dtype=np.int32)
        
    # iterate train set
    for inputs, labels in iter(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        for label, output in zip(labels, outputs.cpu().detach().numpy()):
            confusion[np.argmax(label), np.argmax(output)] += 1

    # compute some metrics
    total = np.sum(confusion)
    accuracy = np.sum(np.diag(confusion)) / total
    per_class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)
    
    if train_mode:
        model.train()

    return accuracy, per_class_accuracy, confusion


def print_test_results(model):

    '''
    - Computes metrics: accuracy, per class accuracy, confusion matrix
    - Prints these metrics to the console
    '''

    accuracy, per_class_accuracy, confusion = test_model(model)

    print(f'Global accuracy {accuracy:.2%}')
    print('Confusion matrix:'), print(confusion)
    print('Per class accuracies:')
    for acc, name in zip(per_class_accuracy, targets):
        print(f'{name:>12}: {acc:.2%}')


def train_model(model):
    
    '''
    - Handles model training mode
    - Trains the model with parsed arguments
    - Logs training process in wandb
    '''

    model.train()
    
    # construct name
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}-{optimizer_name}-lr{args.lr}'
    
    # init weights and biases
    with wandb.init(project='DLL-Project', name=run_name) as run:
        
        # log some info
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(model)
        
        # progress bar
        mb = master_bar(range(args.epochs))
        
        for epoch in mb:
            
            for inputs, labels in progress_bar(iter(train_loader), parent=mb):

                # move the data to the GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log the loss
                run.log({'loss': loss})
                
            # evaluate the model
            accuracy, per_class_accuracy, confusion = test_model(model)
            mb.main_bar.comment = f'val acc:{accuracy}'

            # log the data
            run.log({'accuracy': accuracy, 'epoch': epoch})


train_model(model)
print_test_results(model)

if args.model == 'lstm':
    torch.save(model.state_dict(), 'state_dict_lstm.pth')

#outputs = model.forward(x_test)
#loss = criterion(outputs, y_test)
#print(loss)
