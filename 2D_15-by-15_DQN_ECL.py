# Problem definition:
# Placing 5 production well sequentially in 2D 15-by-15 heterogeneous reservoir
# Period of well placement is 120 days, and total production period is 600 days (Well placement time from simulation starts: 0-120-240-360-480)
# 다 작성되면 우성이 코드 보내주기
import copy

######################################## 1. Import required modules #############################################
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

def main():
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
    args.master_directory = os.getcwd()
    args.basicfilepath = 'data'
    args.simulation_directory = 'simulation'
    args.variable_save_directory = 'variables'
    args.deeplearningmodel_save_directory = 'model'
    args.ecl_filename = '2D_ECL'
    args.perm_filename = 'PERMX'
    args.well_filename = 'WELL'

    # args.total_episode = 100
    args.learning_rate = 0.1  # Learning rate Alpha
    args.boltzmann_tau_start = 5.0  # Start value of Temperature parameter at Boltzmann policy, Tau
    args.boltzmann_tau_end = 0.1  # End value of Temperature parameter at Boltzmann policy, Tau
    # args.total_reward = 0
    args.epsilon = 0.1

    # # For Implementation
    # args.max_iteration = 50 # Maximum iteration num. of algorithm, MAX_STEPS
    # args.sample_num_per_iter = 50 # Simulation sample num. of each iteration of algorithm
    # args.experience_num_per_iter = 250 # Experience sample num. of each iteration of algorithm, h
    # args.replay_batch_num = 16 # Replay batch num., B
    # args.nn_update_num = 20 # CNN update number, U: [(1) Constant num. of iteration], (2) Lower limit of loss function value
    # args.batch_size = 32 # Batch size, N
    # args.replay_memory_size = 1000 # Replay memory size, K

    # For Debugging
    args.max_iteration = 5  # Maximum iteration num. of algorithm, MAX_STEPS
    args.sample_num_per_iter = 3  # Simulation sample num. of each iteration of algorithm
    args.experience_num_per_iter = 15  # Experience sample num. of each iteration of algorithm, h
    args.replay_batch_num = 4  # Replay batch num., B
    args.nn_update_num = 4  # CNN update number, U: [(1) Constant num. of iteration], (2) Lower limit of loss function value
    args.batch_size = 8  # Batch size, N
    args.replay_memory_size = 30  # Replay memory size, K

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

    # 2023-05-02: For reproduction
    args.random_seed = 202022673
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # For calculation of revenue at each time step
    args.oil_price = 60  # $/bbl
    args.water_treatment = 3  # $/bbl
    args.water_injection = 5  # $/bbl

    # State: Pressure distribution, Oil saturation, Well placement map

    # Action: Well placement (Coordinate of well location)

    # Environment: Reservoir simulator

    # Reward: NPV at each time segment

    args.discount_rate = 0.1  # Used for calculation of NPV
    args.discount_factor = 1  # Used for Q-value update

    # Data for State
    args.input_flag = ('PRESSURE', 'SOIL', 'Well_placement')

    ######################################## 3. Run algorithm #############################################
    # Directory setting
    if not os.path.exists(args.simulation_directory):
        print('Simulation directory does not exist: Created Simulation directory\n')
        os.mkdir(args.simulation_directory)

    if not os.path.exists(args.variable_save_directory):
        print('Variable storage directory does not exist: Created Variable storage directory\n')
        os.mkdir(args.variable_save_directory)

    if not os.path.exists(args.deeplearningmodel_save_directory):
        print('Deep learning model storage directory does not exist: Created Deep learning model storage directory\n')
        os.mkdir(args.deeplearningmodel_save_directory)

    # Implementation of DQN Algorithm
    # Initialize Deep Q Network
    Deep_Q_Network = DQN(args=args, block=BasicBlock).to('cuda')

    # Experience sample queue or Replay memory (double-ended queue, deque)
    replay_memory = Experience_list(args=args)

    optimizer = optim.AdamW(Deep_Q_Network.parameters(), lr=args.learning_rate, amsgrad=True)

    args.tau = args.boltzmann_tau_start

    for m in range(1, args.max_iteration + 1):
        # Generate well placement simulation sample list, length of list is (args.sample_num_per_iter).
        simulation_sample = []

        # Total num. of experience == args.sample_num_per_iter * (args.total_production_time / args.time_step)
        # For this case, Total num. of experience == 50 * (600/120) == 250
        for i in range(1, args.sample_num_per_iter + 1):
            simulation_sample.append(
                _simulation_sampler(args=args, algorithm_iter_count=m, sample_num=i, network=Deep_Q_Network))

        experience_sample = _experience_sampler(args=args, simulation_sample_list=simulation_sample)

        for i in range(0, len(experience_sample)):
            if len(replay_memory) == args.replay_memory_size:
                replay_memory.exp_list.popleft()
            replay_memory.exp_list.append(experience_sample[i])

        for b in range(1, args.replay_batch_num + 1):
            # # Extract b-th experience data from replay memory
            # target_Q = deque()  # Target Q value, yi
            target_Q = []  # Target Q value, yi
            # current_Q = deque()  # For calculation of loss
            current_Q = []  # For calculation of loss
            # next_Q = deque() # For calculation of Target Q value
            next_Q = []  # For calculation of Target Q value

            # We want to know indices of Experience samples which are selected, but DataLoader does not support fuction
            # of searching indices of selected Experience samples directly.
            # Thus, (1) Select Experience samples in replay_memory and (2) Perform DQN training with selected subset of Experience samples
            exp_idx = [random.randint(0, len(replay_memory)-1) for r in range(args.batch_size)] # Indices for subset
            subset_current = Experience_list(args=args)
            for i in range(args.batch_size):
                # subset_current.exp_list.append(replay_memory[exp_idx[i]])
                subset_current.exp_list.append(replay_memory.exp_list[exp_idx[i]])
            subset_next = copy.deepcopy(subset_current) # For calculation of max. Q-value at next state
            for element in subset_next.exp_list: # Replace current state with next state, so if DataLoader called for subset_next, next state will be used for DQN.
                element.current_state = element.next_state
            batch_current = DataLoader(dataset=subset_current, batch_size=args.batch_size, shuffle=False)
            batch_next = DataLoader(dataset=subset_next, batch_size=args.batch_size, shuffle=False)

            for u in range(1, args.nn_update_num + 1):
                for sample_current, sample_next in zip(batch_current, batch_next):
                    # To get related data by searching replay_memory with exp_list, for loop was used.
                    # Maximum Q-value at next_state for each Experience sample in batch only can be calculated as batch unit! (not Experience sample unit!)
                    # Output dimension: (batch_size, 1, gridnum_y, gridnum_x)
                    next_Q_map = Deep_Q_Network.forward(sample_next) # Tensor >> Tensor

                    # Do Well placement masking before finding max. Q-value
                    # next_Q_map_mask = numpy.squeeze(deepcopy(next_Q_map).numpy())
                    # next_Q_map_mask = numpy.squeeze(deepcopy(next_Q_map).numpy(), axis=1) # For 2-D well placement
                    next_Q_map_mask = numpy.squeeze(deepcopy(next_Q_map.detach()).cpu().numpy(), axis=1)  # For 2-D well placement
                    for i in range(args.batch_size):
                        for row in range(len(replay_memory.exp_list[exp_idx[i]].next_state[2])): # replay_memory.exp_list[exp_idx[i]].next_state[2]: Well placement map
                            for col in range(len(replay_memory.exp_list[exp_idx[i]].next_state[2][row])):
                                if replay_memory.exp_list[exp_idx[i]].next_state[2][row][col] == 1:
                                    next_Q_map_mask[i][row][col] = np.NINF # (x, y) for ECL, (Row(y), Col(x)) for Python / 2D-map array

                    for i in range(args.batch_size):
                        # Q-value for current_state will always be used, but Q-value for next_state cannot be used if next_state is terminal state
                        # Output dimension: (batch_size, 1, gridnum_y, gridnum_x)
                        # (x, y) for ECL, (Row=y, Col=x) for Python [Index: 1~nx for ECL, 0~(nx-1) for Python] / 2D-map array
                        current_Q.append(Deep_Q_Network.forward(sample_current)[i][0][int(subset_current.exp_list[i].current_action[1])-1][int(subset_current.exp_list[i].current_action[0])-1].reshape(1))
                        # max_action = max(Q at state s')
                        max_row, max_col = np.where(np.array(next_Q_map_mask[i]) == max(map(max, np.array(next_Q_map_mask[i]))))
                        # next_Q.append(next_Q_map_mask[i][max_row][max_col])
                        next_Q.append(next_Q_map_mask[i][max_row[0]][max_col[0]])

                        # if well_num == 5 (terminal state):
                        #   yi = ri
                        # if np.cumsum(np.array(replay_memory.exp_list[exp_idx[i]].next_state[2])) == 5: # sample.next_state[2]: Well placement map
                        if np.cumsum(replay_memory.exp_list[exp_idx[i]].next_state[2].detach().cpu().numpy())[-1] == 5:  # sample.next_state[2]: Well placement map
                            target_Q.append(replay_memory.exp_list[exp_idx[i]].reward.reshape(1))

                        # elif well_num < 5 (non-terminal state):
                        #   yi = ri + args.discount_factor * max.Q_value(Q_network(s', a'))
                        # elif np.cumsum(np.array(replay_memory.exp_list[exp_idx[i]].next_state[2])) < 5: # sample.next_state[2]: Well placement map
                        elif np.cumsum(replay_memory.exp_list[exp_idx[i]].next_state[2].detach().cpu().numpy())[-1] < 5:  # sample.next_state[2]: Well placement map
                            target_Q.append((replay_memory.exp_list[exp_idx[i]].reward + args.discount_factor * (next_Q[i])).reshape(1))

                # Loss calculation (Mean Square Error (MSE)): L(theta) = sum((yi - Q_network(s, a))^2) / args.batch_size
                criterion = nn.SmoothL1Loss()
                # 'collections.deque' object has no attribute 'size'
                # loss = criterion(target_Q, current_Q)
                loss = criterion(torch.cat(target_Q), torch.cat(current_Q))
                # Update Q-network parameter: theta = theta - args.learning_rate * grad(L(theta))
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward(retain_graph=True)
                optimizer.step()
        # Decrease tau (temperature parameter of Boltzmann policy)
        args.tau = ((args.boltzmann_tau_start - args.boltzmann_tau_end) * np.log(args.max_iteration + 1 - m) + args.boltzmann_tau_end) / ((args.boltzmann_tau_start - args.boltzmann_tau_end) * np.log(args.max_iteration) + args.boltzmann_tau_end) * 5

################################################################################################
###################################### Definition: Class #######################################
################################################################################################

################################### Class: Placement sample #################################### Certified
# WellPlacementSample Class contains information of One full sequence of simulation sample.
class WellPlacementSample:
    def __init__(self, args):
        self.args = args
        self.well_loc_map = [[[0 for i in range(0, args.gridnum_x)] for j in range(0, args.gridnum_y)]]
        self.well_loc_list = []
        self.PRESSURE_map = [[[args.initial_PRESSURE for i in range(0, args.gridnum_x)] for j in range(0, args.gridnum_y)]]
        self.SOIL_map = [[[args.initial_SOIL for i in range(0, args.gridnum_x)] for j in range(0, args.gridnum_y)]]
        self.income = []

################################### Class: Experience sample ################################### Certified
# Experience Class contains One set of (s, a, r, s').
class Experience:
    def __init__(self, args):
        self.args = args
        self.current_state = list
        self.current_action = None
        self.reward = None
        self.next_state = list

    def __transform__(self):
        self.current_state = torch.tensor(data=self.current_state, dtype=torch.float, device=self.args.device, requires_grad=False)
        self.current_action = torch.tensor(data=self.current_action, dtype=torch.float, device=self.args.device, requires_grad=False)
        self.reward = torch.tensor(data=self.reward, dtype=torch.float, device=self.args.device, requires_grad=False)
        self.next_state = torch.tensor(data=self.next_state, dtype=torch.float, device=self.args.device, requires_grad=False)

    def transform(self):
        self.__transform__()


###################### Class: Experience sample (Dataset for DataLoader) ####################### Certified
class Experience_list(Dataset):
    def __init__(self, args):
        self.args = args
        self.exp_list = deque()

    def __len__(self):
        return len(self.exp_list)

    def __getitem__(self, idx):
        return self.exp_list[idx].current_state

################################################################################################
################################## Definition: CNN Structure ###################################
################################################################################################

######################################### CNN Structure ########################################
# Modification of ResNet structure
# https://cryptosalamander.tistory.com/156
class BasicBlock(nn.Module): # Residual block
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        self.conv1.apply(self._init_weight)
        self.conv2.apply(self._init_weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out

    def _init_weight(self, layer, init_type="Xavier"):
        if isinstance(layer, nn.Conv2d):
            if init_type == "Xavier":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif init_type == "He":
                torch.nn.init.kaiming_uniform_(layer.weight)


class DQN(nn.Module):
    def __init__(self, args, block):
        '''
        이 두 코드는 형태가 조금 다르다.
        == super(MyModule,self).__init__()
        == super().__init__()
        super 안에 현재 클래스를 명시해준 것과 아닌 것으로 나눌 수 있는데 이는 기능적으론 아무런 차이가 없다.
        파생클래스와 self를 넣어서 현재 클래스가 어떤 클래스인지 명확하게 표시 해주는 용도이다.
        super(파생클래스, self).__init__()
        '''
        '''
        Structure of DQN
        (1) input (Dim: (batch_size, len(state), gridnum_y, gridnum_x))
        (2) Conv1-BatchNorm-ReLU (Dim: (batch_size, 48, gridnum_y, gridnum_x))
        (3) Residual block (Dim: (batch_size, 48, gridnum_y, gridnum_x))
        (4) Conv2-BatchNorm-ReLU (Dim: (batch_size, 24, gridnum_y, gridnum_x))
        (5) Conv3 (Dim: (batch_size, 1, gridnum_y, gridnum_x)) << Objective: Q-value of each action at current state
        Action masking will be done by modified Boltzmann policy
        '''
        super(DQN, self).__init__()
        if args.input_flag: self.num_of_channels = len(args.input_flag)
        else: self.num_of_channels = 3

        self.in_planes = 0
        self.out_channel = 48

        self.conv1 = nn.Conv2d(in_channels=self.num_of_channels, out_channels=self.out_channel, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(self.out_channel)

        self.layer1 = self.make_layer(block=block, out_planes=self.out_channel, num_blocks=1, stride=1)

        self.conv2 = nn.Conv2d(in_channels=self.out_channel, out_channels=round(self.out_channel/2), kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(round(self.out_channel/2))

        self.conv3 = nn.Conv2d(in_channels=round(self.out_channel/2), out_channels=1, kernel_size=3, stride=1, padding='same')

        self.conv1.apply(self._init_weight)
        self.conv2.apply(self._init_weight)
        self.conv3.apply(self._init_weight)

    def make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # out = self.layer1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        return out

    # Weight initializer method
    def _init_weight(self, layer, init_type="Xavier"):
        if isinstance(layer, nn.Conv2d):
            if init_type == "Xavier":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif init_type == "He":
                torch.nn.init.kaiming_uniform_(layer.weight)


################################################################################################
##################################### Definition: Function #####################################
################################################################################################

############################## Reading Eclipse Dynamic Data (.PRT) ############################# Certified
# algorithm_iter_count: algorithm iteration num. (m)
# sample_num: sample num. (1 ~ args.sample_num_per_iter)
# tstep_idx: sample time step index
# filename: simulation file name
# dynamic_type: dynamic data type to collect ('PRESSURE' or 'SOIL')
def _read_ecl_prt_2d(args, algorithm_iter_count: int, sample_num: int, tstep_idx: int, dynamic_type: str) -> list:
    # Check if dynamic type input is (1) 'PRESSURE', (2) 'SOIL'
    if not dynamic_type in ['PRESSURE', 'SOIL']:
        print("Assign correct dynamic data output type!: 'PRESSURE', 'SOIL'")
        return -1

    # File IO
    # 1. Open .PRT file
    # with open(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}", f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx}.PRT")) as file_read:
    with open(f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx}.PRT") as file_read:
        line = file_read.readline()
        if dynamic_type == 'PRESSURE':
            # 2. Find the location of dynamic data (PRESSURE case)
            while not line.startswith(f"  {dynamic_type} AT   {args.time_step * tstep_idx}"):
                line = file_read.readline()
            # 3. Dynamic data starts from 10th line below the line ["  {dynamic_type} AT   {args.time_step * tstep_idx}"]
            for i in range(1,10+1):
                line = file_read.readline()
            # 4. Collect dynamic data
            lines_converted = []
            for i in range(1, args.gridnum_y+1):
                lines_converted.append([element.strip() for element in line.split()][3::])
                line = file_read.readline()
        elif dynamic_type == 'SOIL':
            # 2. Find the location of dynamic data (SOIL case)
            while not line.startswith(f"  {dynamic_type}     AT   {args.time_step * tstep_idx}"):
                line = file_read.readline()
            # 3. Dynamic data starts from 10th line below the line ["  {dynamic_type}     AT   {args.time_step * tstep_idx}"]
            for i in range(1,10+1):
                line = file_read.readline()
            # 4. Collect dynamic data
            lines_converted = []
            for i in range(1, args.gridnum_y+1):
                lines_converted.append([element.strip() for element in line.split()][3::])
                line = file_read.readline()

    # 5. Post-processing (String replacement from (1) '*' to '.', (2) String to Float (Only for 2D)
    for i in range(len(lines_converted)):
        for j in range(len(lines_converted[i])):
            lines_converted[i][j] = float(lines_converted[i][j].replace('*', '.'))

    return lines_converted

################## Reading Eclipse Production or Injection Data (.RSM) ################### Certified
# algorithm_iter_count: algorithm iteration num. (m)
# sample_num: sample num. (1 ~ args.sample_num_per_iter)
# tstep_idx: sample time step index
# filename: simulation file name
# data_type: Production or Injection result data type ('FOPT', 'FWPT', 'FWIT')
def _read_ecl_rsm(args, algorithm_iter_count: int, sample_num: int, tstep_idx: int, dynamic_type: str) -> list:
    # Check if data type input is (1) 'FOPT', (2) 'FWPT', (3) 'FWIT'
    if not dynamic_type in ['FOPT', 'FWPT', 'FWIT']:
        print("Assign correct output data type!: 'FOPT', 'FWPT', 'FWIT'")
        return -1

    # File IO
    # 1. Open .RSM file
    # with open(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}", f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx}.RSM")) as file_read:
    with open(f"{args.ecl_filename}_SAM{sample_num}_SEQ{tstep_idx+1}.RSM") as file_read:
        line = file_read.readline()
        # 2. Find the location of simulation result data
        while not line.startswith(f" TIME"):
            line = file_read.readline()
        # 3. 1st time of simulation result data starts from 6th line below the line [" TIME"]
        for i in range(1,5+1):
            line = file_read.readline()
        # 4. Collect production or injection data
        lines_converted = []
        # if dynamic_type == 'FOPT':
        #     for i in range(1, round(args.total_production_time/args.time_step)+1):
        #         lines_converted.append([element.strip() for element in line.split()][2])
        #         line = file_read.readline()
        # elif dynamic_type == 'FWPT':
        #     for i in range(1, round(args.total_production_time/args.time_step)+1):
        #         lines_converted.append([element.strip() for element in line.split()][3])
        #         line = file_read.readline()
        # elif dynamic_type == 'FWIT':
        #     for i in range(1, round(args.total_production_time/args.time_step)+1):
        #         lines_converted.append([element.strip() for element in line.split()][4])
        #         line = file_read.readline()
        if dynamic_type == 'FOPT':
            while line:
                lines_converted.append([element.strip() for element in line.split()][2])
                line = file_read.readline()
        elif dynamic_type == 'FWPT':
            while line:
                lines_converted.append([element.strip() for element in line.split()][3])
                line = file_read.readline()
        elif dynamic_type == 'FWIT':
            while line:
                lines_converted.append([element.strip() for element in line.split()][4])
                line = file_read.readline()

    # 5. Post-processing (String replacement from (1) '*' to '.', (2) String to Float)
    for i in range(len(lines_converted)):
        lines_converted[i] = float(lines_converted[i].replace('*', '.'))

    return lines_converted

################################### Run ECL Simulator #################################### Certified
# program: 'eclipse' or 'frontsim'
# filename: simulation file name (ex. 2D_ECL_Sam1_Seq2)
def _run_program(args, program: str, filename: str):
    # Check if dynamic type input is (1) 'eclipse', (2) 'frontsim'
    if not program in ['eclipse', 'frontsim']:
        print("Use correct simulator exe file name!: 'eclipse', 'frontsim'")
        return -1
    command = fr"C:\\ecl\\2009.1\\bin\\pc\\{program}.exe {filename} > NUL"
    os.system(command)

#################################### Boltzmann policy #################################### Certified
# Transform Q-value map (2D array) to Well placement probability map (2D array)
def _Boltzmann_policy(args, Q_value: list, well_placement: list) -> list:
    Q_value_list = np.squeeze(Q_value)
    exp_tau = deepcopy(Q_value_list)
    probability = deepcopy(Q_value_list)

    # Prevent overflow error by exponential operation
    max_Q_value = np.array(Q_value_list).flatten().max()

    # Get exponential of all elements in Q_value
    for i in range(0, args.gridnum_y):
        for j in range(0, args.gridnum_x):
            exp_tau[i][j] = np.exp((exp_tau[i][j]-max_Q_value)/args.tau)

    # Calculate probability map
    for i in range(0, args.gridnum_y):
        for j in range(0, args.gridnum_x):
            probability[i][j] = exp_tau[i][j] / np.concatenate(np.array(exp_tau)).sum()

    # Mask probability map: Setting probability = 0 where wells were already exists,
    # and Scale the rest of probability map
    probability = [[0 if well_placement[i][j] != 0 else probability[i][j] for j in range(args.gridnum_x)] for i in range(args.gridnum_y)]
    probability = [[(probability[i][j]/np.concatenate(np.array(probability)).sum()) for j in range(args.gridnum_x)] for i in range(args.gridnum_y)]

    return probability

#################################### Action Selection #################################### Certified
# Select well location from well placement probability map
def _select_well_loc(args, probability: list) -> tuple:
    # Create cumulative probability function with given probability map by policy
    cumsum_prob = np.cumsum(probability)
    CDF = np.append([0], cumsum_prob)

    # Generate random number (0~1)
    CDF_prob = random.random()

    # Find corresponding well location
    for i in range(0, len(CDF)-1):
        if (CDF_prob >= CDF[i]) and (CDF_prob < CDF[i+1]):
            well_loc = ((i%args.gridnum_x)+1, (i//args.gridnum_x)+1) # (x, y) for ECL, (Row, Col) for Python.
            return well_loc

    # If well location selection failed.
    print("Well location selection was not appropriately done!")

#################################### NPV Calculation #####################################
def _calculate_income(args, tstep_idx: int, FOPT: list, FWPT: list, FWIT: list) -> float:
    # Calculate income from [tstep_idx] to [tstep_idx+1]
    # e.g. tstep_idx == 0 >> income of 0 ~ 120 day, tstep_idx == 1 >> income of 120 ~ 240 day
    oil_income = (FOPT[tstep_idx+1] - FOPT[tstep_idx]) * args.oil_price / (((1 + args.discount_rate)) ** (args.time_step * (tstep_idx + 1) / 365))
    water_treat = (FWPT[tstep_idx+1] - FWPT[tstep_idx]) * args.water_treatment / (((1 + args.discount_rate)) ** (args.time_step * (tstep_idx + 1) / 365))
    water_inj = (FWIT[tstep_idx+1] - FWIT[tstep_idx]) * args.water_injection / (((1 + args.discount_rate)) ** (args.time_step * (tstep_idx + 1) / 365))

    # 2023-07-07: Changed unit income ($ >> MM$)
    # income = oil_income - water_treat - water_inj
    income = oil_income - water_treat - water_inj
    # income = income / (10^6)

    return income

############################ Generating Simulation Data File #############################
# Need to utilize RESTART Option
def _ecl_data_generate(args, algorithm_iter_count: int, sample_num: int, timestep: int, well_loc_list: list[tuple]) -> None:
    output_data_file = []
    output_perm_file = []
    output_well_file = []

    # Read and modify simulation data template on Python
    with open(f"{os.path.join(args.basicfilepath, args.ecl_filename)}.template", 'r') as file_read_data:
        line = file_read_data.readline()
        output_data_file.append(line)
        while not line.startswith("[#PERMX]"):
            line = file_read_data.readline()
            output_data_file.append(line)
        line = line.replace("[#PERMX]", f"\'{args.perm_filename}_Sam{sample_num}_Seq{timestep}.DATA\'")
        output_data_file[-1] = line
        while not line.startswith("[#WELL]"):
            line = file_read_data.readline()
            output_data_file.append(line)
        line = line.replace("[#WELL]", f"\'{args.well_filename}_Sam{sample_num}_Seq{timestep}.DATA\'")
        output_data_file[-1] = line
        while line:
            line = file_read_data.readline()
            output_data_file.append(line)

    # Read Permeability distribution file
    with open(f"{os.path.join(args.basicfilepath, args.perm_filename)}.DATA", 'r') as file_read_perm:
        line = file_read_perm.readline()
        output_perm_file.append(line)
        while line:
            line = file_read_perm.readline()
            output_perm_file.append(line)

    # Write simulation main data and include files
    sample_simulation_directory = f"Step{algorithm_iter_count}_Sample{sample_num}"
    sample_data_name = f"{args.ecl_filename}_Sam{sample_num}_Seq{timestep}.DATA"
    sample_perm_name = f"{args.perm_filename}_Sam{sample_num}_Seq{timestep}.DATA"
    sample_well_name = f"{args.well_filename}_Sam{sample_num}_Seq{timestep}.DATA"

    with open(f"{os.path.join(args.simulation_directory, sample_simulation_directory, sample_data_name)}", 'w') as file_write_data:
        for i in range(len(output_data_file)):
            file_write_data.write(output_data_file[i])

    with open(f"{os.path.join(args.simulation_directory, sample_simulation_directory, sample_perm_name)}", 'w') as file_write_perm:
        for i in range(len(output_perm_file)):
            file_write_perm.write(output_perm_file[i])

    for i in range(len(well_loc_list)):
        output_well_file.append(f"--WELL #{i+1}\n"
                                f"WELSPECS\n P{i+1} ALL {well_loc_list[i][0]} {well_loc_list[i][1]} 1* LIQ 3* NO /\n/\n \n"
                                f"COMPDAT\n P{i+1} {well_loc_list[i][0]} {well_loc_list[i][1]} 1 1 1* 1* 1* 1 1* 1* 1* Z /\n/\n \n"
                                f"WCONPROD\n P{i+1} 1* BHP 5000 4* 1500.0 /\n/\n \n"
                                f"TSTEP\n 1*{args.time_step} /\n \n \n")

    with open(f"{os.path.join(args.simulation_directory, sample_simulation_directory, sample_well_name)}", 'w') as file_write_well:
        for i in range(len(output_well_file)):
            file_write_well.write(output_well_file[i])

################################### Simulation Sampler ###################################
# Make One full Well placement sample
def _simulation_sampler(args, algorithm_iter_count: int, sample_num: int, network) -> WellPlacementSample:
    well_placement_sample = WellPlacementSample(args=args)

    Q_network = network

    if not os.path.exists(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}")):
        os.mkdir(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}"))

    # Well placement sampling
    for time_step in range(0, args.total_well_num_max):
        ####################################################################################################################################################################################
        # Inference of Q-value from PRESSURE, SOIL, and Well placement
        # Q_map = Q_network.forward([well_placement_sample.PRESSURE_map[time_step], well_placement_sample.SOIL_map[time_step], well_placement_sample.well_loc_map[time_step]])
        # Q_map = Q_network.forward(torch.tensor(data = [well_placement_sample.PRESSURE_map[time_step], well_placement_sample.SOIL_map[time_step], well_placement_sample.well_loc_map[time_step]], dtype=torch.float, device='cuda', requires_grad=True))
        Q_map = Q_network.forward(torch.tensor(data = [[well_placement_sample.PRESSURE_map[time_step], well_placement_sample.SOIL_map[time_step], well_placement_sample.well_loc_map[time_step]]], dtype=torch.float, device='cuda', requires_grad=True))

        ####################################################################################################################################################################################

        # Calculate well placement probability and Specify well location
        prob = _Boltzmann_policy(args=args, Q_value=Q_map.tolist(), well_placement=well_placement_sample.well_loc_map[time_step])
        well_loc = _select_well_loc(args=args, probability=prob)
        well_placement_sample.well_loc_list.append(well_loc)

        # Generate and run simulation file
        _ecl_data_generate(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, timestep=time_step+1, well_loc_list=well_placement_sample.well_loc_list)

        os.chdir(os.path.join(args.simulation_directory, f"Step{algorithm_iter_count}_Sample{sample_num}"))
        _run_program(args=args, program='eclipse', filename=f"{args.ecl_filename}_Sam{sample_num}_Seq{time_step+1}")

        # Read PRESSURE, SOIL map from .PRT file and calculate income with .RSM file
        pressure_map = _read_ecl_prt_2d(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='PRESSURE')
        soil_map = _read_ecl_prt_2d(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='SOIL')

        # fopt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='FOPT')
        # fwpt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='FWPT')
        # fwit = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step+1, dynamic_type='FWIT')
        fopt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FOPT')
        fwpt = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FWPT')
        fwit = _read_ecl_rsm(args=args, algorithm_iter_count=algorithm_iter_count, sample_num=sample_num, tstep_idx=time_step, dynamic_type='FWIT')

        income = _calculate_income(args=args, tstep_idx=time_step, FOPT=fopt, FWPT=fwpt, FWIT=fwit)

        well_placement_map = deepcopy(well_placement_sample.well_loc_map[time_step])
        for i in range(0, args.gridnum_x):
            for j in range(0, args.gridnum_y):
                if (i == well_loc[0]-1) and (j == well_loc[1]-1):
                    well_placement_map[j][i] = 1

        # Append PRESSURE map, SOIL map, Well placement map, Income
        well_placement_sample.PRESSURE_map.append(pressure_map)
        well_placement_sample.SOIL_map.append(soil_map)
        well_placement_sample.well_loc_map.append(well_placement_map)
        well_placement_sample.income.append(income)

        os.chdir('../../')

    return well_placement_sample

################################### Experience Sampler ###################################
# Collect and save Experience instances from simulation samples
def _experience_sampler(args, simulation_sample_list: list[WellPlacementSample])-> list[Experience]:
    # 1. Save all Experience instances from simulation samples (experience_list)
    # 2. Pick random experience from experience_list
    experience_list = []

    for i in range(0, args.sample_num_per_iter):
        for j in range(0, args.total_well_num_max):
            exp = Experience(args=args)
            # torch.tensor(data=list, dtype=torch.float, device='cuda', requires_grad=True)
            exp.current_state = [simulation_sample_list[i].PRESSURE_map[j], simulation_sample_list[i].SOIL_map[j], simulation_sample_list[i].well_loc_map[j]]
            exp.current_action = simulation_sample_list[i].well_loc_list[j]
            exp.reward = simulation_sample_list[i].income[j]
            exp.next_state = [simulation_sample_list[i].PRESSURE_map[j+1], simulation_sample_list[i].SOIL_map[j+1], simulation_sample_list[i].well_loc_map[j+1]]
            exp.transform()
            experience_list.append(exp)

    experience_sample = random.sample(experience_list, args.experience_num_per_iter)

    return experience_sample

if __name__ == "__main__":
    main()
