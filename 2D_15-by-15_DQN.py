# Problem definition:
# Placing 5 production well sequentially in 2D 15-by-15 heterogeneous reservoir
# Period of well placement is 120 days, and total production period is 600 days (Well placement time from simulation starts: 0-120-240-360-480)
# 다 작성되면 우성이 코드 보내주기

# Import library
import os # For CMG/ECL cmd command prompt execution and Directory generation
import shutil
import random # For reproduction of research
import argparse # For definition of set of arguments
import numpy as np
import itertools
import pickle # For saving simulation results

def main():
    # Define arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.BRUTEFORCE = 1
    args.DQN = 2

    args.search_method = args.BRUTEFORCE
    # args.search_method = args.DQN

    args.total_episode = 100
    args.learning_rate = 0.1
    args.total_reward = 0
    args.epsilon = 0.1

    args.gridnum_x = 15
    args.gridnum_y = 15
    # args.gridnum_x = 2 # For debugging
    # args.gridnum_y = 2 # For debugging
    args.gridsize_x = 120 # ft
    args.gridsize_y = 120 # ft

    args.total_wellnum = 5
    args.time_step = 120 # Day, Duration between each actions
        
    args.random_seed = 202022673
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # State: Pressure and Saturation distribution map

    # Action: Well placement (Coordinate of well location)
    # 2D-like action table
    args.action_set = [[(i, j) for i in range(1, args.gridnum_x+1)] for j in range(1, args.gridnum_y+1)]

    # Deep Q-Network
    pass # Need to be implemented

    # True value table (Cumulative oil production) for Bruteforce search - Dictionary
    # Key: Order of well placement, Value: Cumulative oil production
    # PROBLEM!: Very Large Number of Elements in Dictionary! (225*224*223*222*221 == 551,417,630,400)
    args.True_OilProd = {}
    # for i in sum(args.actions, []):
    #     for j in sum(args.actions, []):
    #         args.True_OilProd[i, j] = 0
    # Should we use Greedy search method for Bruteforce search?

    args.discount_factor = 1

    # Reward: Cumulative oil production after each action

    # 1. Bruteforce search
    if args.search_method == args.BRUTEFORCE:
        '''
        # Directory setting
        # -- Master directory (eg. Method name)
        # ---- Simulation directories (of each well placement)
        # ---- Master simulation data file (.dat)
        # ---- Master simulation include template file (.template.inc)
        '''

        # Can be improved by define function or class of simulator (ECL/CMG)
        # Using CMG Simulator
        args.master_directory = "D:\\Lab_Meeting\\Simulation_Model\\2D_15-by-15\\Simulation\\Bruteforce_search"
        args.simulation_directory = "" # Location of actual simulation of each case happens
        args.simulation_data_file = "" # Location of actual simulation data file
        args.simulation_include_file = "" # Location of actual simulation include file
        args.simulation_log = "" # Location of actual simulation log file
        args.master_simulation_data_file = args.master_directory + "\\2D_15-by-15.dat" # Location of master simulation data file
        args.master_well_placement_template = args.master_directory + "\\2D_15-by-15_WellPlacement.template_1.inc" # Location of master simulation include file
        args.results_report_command_template = args.master_directory + "\\2D_15-by-15_Results_Template.rwd" # Location of master CMG Results Report command file
        args.simulation_results_report_command = "" # Location of actual CMG Results Report command file

        # Saving former well location
        args.former_well_location_string = ""
        args.former_well_location_array = []

        # Making basic simulation directories
        if not os.path.exists(args.master_directory):
            os.makedirs(args.master_directory)

        for well_number in range(1, args.total_wellnum+1): # Finding optimal solution by Greedy search
            args.suboptimal_solution = {} # Saving sub-optimal solution by Greedy search

            for well_location in sum(args.action_set, []):
                if not (well_location in args.former_well_location_array): # If well location was in the former well location record, pass to other well locations                        
                    # Make simulation directory and directory strings
                    # Make sure that there are NO SPACEs in strings
                    if well_number == 1: 
                        args.simulation_directory = args.master_directory + f"\\{well_location}".replace(" ", "")
                    else: 
                        args.simulation_directory = args.master_directory + f"\\{args.former_well_location_string}{well_location}".replace(" ", "")

                    args.master_well_placement_template = args.master_directory + f"\\2D_15-by-15_WellPlacement.template_{well_number}.inc"
                    args.simulation_data_file = args.simulation_directory + "\\2D_15-by-15.dat"
                    args.simulation_include_file = args.simulation_directory + f"\\2D_15-by-15_WellPlacement.template_{well_number}.inc"
                    args.simulation_log = args.simulation_directory + "\\2D_15-by-15.log"
                    args.simulation_results_report_command_template = args.simulation_directory + "\\2D_15-by-15_Results_Template.rwd"
                    args.simulation_results_report_command = args.simulation_directory + "\\2D_15-by-15_Results.rwd"
                    args.simulation_results_report_out = args.simulation_directory + "\\2D_15-by-15_Results.rwo"

                    # # For license failure, just run CMG Results Report and read simulation results
                    if os.path.exists(args.simulation_directory):
                        os.chdir(args.simulation_directory)
                        # os.system(f"\"C:\\Program Files\\CMG\\RESULTS\\2022.30\\Win_x64\\exe\\Report.exe\" {args.simulation_results_report_command} {args.simulation_results_report_out} > NUL")
                        with open(args.simulation_results_report_out, "r") as file_read:
                            lines = file_read.readlines()
                            lines[6] = lines[6].split("\t") # Cumulative oil production after 120 days
                            args.suboptimal_solution[well_location] = lines[6][1]
                        # Get out from simulation directory
                        os.chdir(args.master_directory)
                        continue
                    
                    if not os.path.exists(args.simulation_directory):
                        os.makedirs(args.simulation_directory)

                    # Copy simulation files and (.dat and .template.inc files) from master directory
                    shutil.copyfile(args.master_simulation_data_file, args.simulation_data_file)
                    shutil.copyfile(args.master_well_placement_template, args.simulation_include_file)
                    shutil.copyfile(args.results_report_command_template, args.simulation_results_report_command_template)
                    
                    # Go to simulation directory
                    os.chdir(args.simulation_directory)

                    # Read Main simulation data file and replace well placement include file
                    with open(args.simulation_data_file, 'r') as file_read:
                        lines = file_read.readlines()
                        lines[332] = lines[332].replace("[#IncludeFile]", f"\'2D_15-by-15_WellPlacement.template_{well_number}.inc\'")
                    with open(args.simulation_data_file, "w") as file_write:
                        for i in range(len(lines)):
                            file_write.write(lines[i])

                    # Read well placement template file and replace well placement and coordinate at template file
                    with open(args.simulation_include_file, 'r') as file_read:
                        lines = file_read.readlines()
                        well_data_location = [i for i in range(len(lines)) if "**[#Well_Prod" in lines[i]]
                        lines[well_data_location[0]] = lines[well_data_location[0]].replace(f"**[#Well_Prod_{well_number}]\n", 
                                                                                        f"WELL \'Prod_{well_number}\'\n PRODUCER \'Prod_{well_number}\'\n" + 
                                                                                        "OPERATE MIN BHP 1500 CONT REPEAT\n OPERATE MAX STO 5000 CONT REPEAT\n" +
                                                                                        "**       rad geofrac wfrac skin\n" +
                                                                                        "GEOMETRY K 0.25 0.37 1 0\n" + 
                                                                                        f"PERF GEOA \'Prod_{well_number}\'\n" +
                                                                                        "**       UBA ff Status Connection\n" +
                                                                                        f"{well_location[0]} {well_location[1]} 1 1.0 OPEN FLOW-TO \'SURFACE\'\n"+
                                                                                        f"LAYERXYZ \'Prod_{well_number}\'\n"+
                                                                                        "** perf geometric data: UBA, block entry(x,y,z) block exit(x,y,z), length\n"+
                                                                                        f"{well_location[0]} {well_location[1]} 1 {well_location[0]*args.gridsize_x-args.gridsize_x/2} {well_location[1]*args.gridsize_y-args.gridsize_y/2} 4200 {well_location[0]*args.gridsize_x-args.gridsize_x/2} {well_location[1]*args.gridsize_y-args.gridsize_y/2} 4240 40\n")
                    with open(args.simulation_include_file, "w") as file_write:
                        for i in range(len(lines)):
                            file_write.write(lines[i])

                    # Run simulation
                    simulation_run = os.system(f"\"C:\\Program Files\\CMG\\IMEX\\2022.30\\Win_x64\\EXE\\mx202230.exe\" -wd {args.simulation_directory} -dimsum -f {args.simulation_data_file} -log {args.simulation_log}")
                    if simulation_run != 0:
                        raise Exception("Simulation execution was not occured!")

                    # Write CMG Results Report command file 
                    with open(args.simulation_results_report_command_template, 'r') as file_read:
                        lines = file_read.readlines()
                        lines[4] = lines[4].replace("[#TIME]", f"{well_number*args.time_step}")
                        lines[9] = lines[9].replace("[#TIME]", f"{well_number*args.time_step}")
                        lines[10] = lines[10].replace("[#TIME]", f"{well_number*args.time_step}")
                        with open(args.simulation_results_report_command, "w") as file_write:
                            for i in range(len(lines)):
                                file_write.write(lines[i])

                    # Run CMG Results Report command file and read results (Cumulative oil production at specific time)
                    os.system(f"\"C:\\Program Files\\CMG\\RESULTS\\2022.30\\Win_x64\\exe\\Report.exe\" {args.simulation_results_report_command} {args.simulation_results_report_out} > NUL")
                    with open(args.simulation_results_report_out, "r") as file_read:
                        lines = file_read.readlines()
                        lines[6] = lines[6].split("\t") # Cumulative oil production after 120 days
                        args.suboptimal_solution[well_location] = lines[6][1]
                    # Get out from simulation directory
                    os.chdir(args.master_directory)
                    
                    # print(args.suboptimal_solution)

                    # Saving the simulation results
                    # with open("Simulation_Results_BruteForce.pkl", "w") as file_result:
                    #     pickle.dump(args.True_OilProd, file_result)

                else: # If well location was in the former well location record, pass to other well locations
                    continue

            # Find sub-optimal well placement solution
            sub_optimal_well_location = max(args.suboptimal_solution, key=args.suboptimal_solution.get)
            args.former_well_location_string += f"{sub_optimal_well_location}".replace(" ", "")
            args.former_well_location_array.append(sub_optimal_well_location)
            print(args.suboptimal_solution)
            print(f"Sub optimal well placement at {well_number} iteration: ", sub_optimal_well_location, f", Cumulative oil prod at {sub_optimal_well_location}: ", args.suboptimal_solution[sub_optimal_well_location])
            print("\n")

            # Modify Well placement include file to seek next sub-optimal well placement
            if not well_number == args.total_wellnum:
                args.master_well_placement_template = args.master_directory + f"\\2D_15-by-15_WellPlacement.template_{well_number+1}.inc"
                with open(args.master_well_placement_template, 'r') as file_read:
                    lines = file_read.readlines()
                    for i in range(1, well_number+1):
                        well_data_location = [j for j in range(len(lines)) if "**[#Well_Prod" in lines[j]]
                        lines[well_data_location[0]] = lines[well_data_location[0]].replace(f"**[#Well_Prod_{i}]\n", 
                                                                                        f"WELL \'Prod_{i}\'\n PRODUCER \'Prod_{i}\'\n" + 
                                                                                        "OPERATE MIN BHP 1500 CONT REPEAT\n OPERATE MAX STO 5000 CONT REPEAT\n" +
                                                                                        "**       rad geofrac wfrac skin\n" +
                                                                                        "GEOMETRY K 0.25 0.37 1 0\n" + 
                                                                                        f"PERF GEOA \'Prod_{i}\'\n" +
                                                                                        "**       UBA ff Status Connection\n" +
                                                                                        f"{args.former_well_location_array[i-1][0]} {args.former_well_location_array[i-1][1]} 1 1.0 OPEN FLOW-TO \'SURFACE\'\n"+
                                                                                        f"LAYERXYZ \'Prod_{i}\'\n"+
                                                                                        "** perf geometric data: UBA, block entry(x,y,z) block exit(x,y,z), length\n"+
                                                                                        f"{args.former_well_location_array[i-1][0]} {args.former_well_location_array[i-1][1]} 1 {args.former_well_location_array[i-1][0]*args.gridsize_x-args.gridsize_x/2} {args.former_well_location_array[i-1][1]*args.gridsize_y-args.gridsize_y/2} 4200 {args.former_well_location_array[i-1][0]*args.gridsize_x-args.gridsize_x/2} {args.former_well_location_array[i-1][1]*args.gridsize_y-args.gridsize_y/2} 4240 40\n")
                with open(args.master_well_placement_template, "w") as file_write:
                    for i in range(len(lines)):
                        file_write.write(lines[i])
            else:
                continue

    
    # 2. DQN approach
    elif args.search_method == args.DQN:
        '''
        # Directory setting
        # -- Master directory (eg. Method name)
        # ---- Simulation directories (of each well placement)
        # ---- Master simulation data file (.dat)
        # ---- Master simulation include template file (.template.inc)
        '''
        args.simulation_directory = "D:\\Lab Meeting\\Simulation Model\\2D_5-by-5\\Simulation\\Q-Learning"
        args.simulation_template = args.simulation_directory + "\\2D_5-by-5.template"

        # Temperature parameter at Boltzmann policy
        # Big tau == more exploration
        # Small tau == less exploration
        args.tau = 5


        # Using Decaying epsilon greedy method
        # Print all Q-values 
        for i in range(args.total_episode):
            # Define current state

            print(f'========== Episode {i+1} / {args.total_episode} started ==========')

            # 

            print(f'%%%%%%%%%% Episode {i+1} / {args.total_episode} finished %%%%%%%%%%\n')

    else:
        pass

    # 3. Comparision between Bruteforce search and DQN approach

if __name__ == "__main__":
    main()