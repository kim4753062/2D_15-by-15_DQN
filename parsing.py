######################################## 1. Import required modules #############################################
import copy
import numpy as np
from collections import deque
import os.path
import shutil
import pickle
import dill

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import numpy.random
import torch.random

from datetime import datetime

from copy import deepcopy

import warnings
warnings.filterwarnings(action='ignore')

######################################## 2. Define arguments #############################################
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

# 2023-05-02
# For Using GPU (CUDA)
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2023-05-02
# For using tensorboard
args.istensorboard = False

'''
Problem definition:
Placing 5 production well sequentially in 2D 15-by-15 heterogeneous reservoir
Period of well placement is 120 days, and total production period is 600 days (Well placement time from simulation starts: 0-120-240-360-480)
'''

'''
Deep Q Network (DQN) State, Action, Environment, Reward definition:

State: Pressure distribution, Oil saturation, Well placement map
Action: Well placement (Coordinate of well location)
Environment: Reservoir simulator
Reward: NPV at each time segment
'''

'''
Directory setting: ([]: Folder)

- [Master directory] (Prerequisite directory)
--- Algorithm launcher code (.py, .ipynb, ...) (Prerequisite file)

--- [Basic simulation data directory] (data) (Prerequisite directory)
----- Simulation data template (Current simulator type: Eclipse, .DATA) (Prerequisite file)
----- Simulation permeability set file (.mat, .DATA, ...) (Prerequisite file)

--- [Simulation directory] (simulation)
----- [Simulation sample directory #f"Step{num. of algorithm iteration}_Sample{sample number}"]
------- Simulation data file (.DATA): for each Well placement timestep
------- Simulation include file (PERMX.DATA, WELL.DATA)
------- # File naming convention: f"{file type}_Sam{sample number}_Seq{timestep index}.DATA", Sam: Sample number (starts from 1), Seq: Time sequence (starts from 1)

--- [Variable storage directory] (Variables)
----- Variable storage.pkl
----- Global variable storage.dill

--- [Deep learning model storage directory] (DL Model)
----- Deep learning model.pkl
'''

# Modified from J.Y. Kim. (2020)
# Arguments: Directory and File name
args.master_directory = os.getcwd()
args.basicfilepath = 'data'
args.simulation_directory = 'simulation'
args.variable_save_directory = 'variables'
args.deeplearningmodel_save_directory = 'model'
args.ecl_filename = '2D_ECL'
args.perm_filename = 'PERMX'
args.well_filename = 'WELL'

# Arguments: Reservoir simulation
args.gridnum_x = 15
args.gridnum_y = 15
args.gridsize_x = 120  # ft
args.gridsize_y = 120  # ft

args.time_step = 120  # days
args.total_production_time = 600  # days

args.prod_well_num_max = 5
args.inj_well_num_max = 0
args.total_well_num_max = args.prod_well_num_max + args.inj_well_num_max

args.initial_PRESSURE = 3500  # psi
args.initial_SOIL = 0.75

# Arguments: Random seed number
args.random_seed = 202022673
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# Arguments: Price and Cost of oil and water
args.oil_price = 60  # $/bbl
args.water_treatment = 3  # $/bbl
args.water_injection = 5  # $/bbl

# Arguments: Hyperparameters for Deep Q Network (DQN)
args.learning_rate = 0.001  # Learning rate Alpha
args.boltzmann_tau_start = 8.0  # Start value of Temperature parameter at Boltzmann policy, Tau
args.boltzmann_tau_end = 0.1  # End value of Temperature parameter at Boltzmann policy, Tau
args.boltzmann_tau_tracker = [args.boltzmann_tau_start]
args.epsilon = 0.1

args.max_iteration = 20  # Maximum iteration num. of algorithm, MAX_STEPS

args.sample_num_per_iter = 20  # Simulation sample num. of each iteration of algorithm
args.experience_num_per_iter = args.total_well_num_max * args.sample_num_per_iter  # Experience sample num. of each iteration of algorithm, h

args.replay_batch_num = 16  # Replay batch num., B

args.nn_update_num = 1  # CNN update number, U: [(1) Constant num. of iteration], (2) Lower limit of loss function value

args.batch_size = 16  # Batch size, N

args.replay_memory_size = 2000  # Replay memory size, K

args.discount_rate = 0.1  # Used for calculation of NPV
args.discount_factor = 1  # Used for Q-value update

args.input_flag = ('PRESSURE', 'SOIL', 'Well_placement')  # Data for State