Keaton Spiller Tue Doan Long Chung Nahom Ketema
CS441

# Othello
Othello MiniMax with three Heuristics and RL Neural Network

RL Neural Network
﻿Requirements 
* Python 3.10 or newer version
* Required file q_weights.csv to store neural network weights
* Put AI_Final_project_RL.py and q_weights.csv at the same folder
The program can be executed by run the following command
        > AI_Final_project_RL.py
After that, follow the guidelines on the screen to select a right option

MiniMax
How to Run the MiniMax algorithm for three different heuristics

heuristic1 = 'number_of_flipped_tiles' 
heuristic2 = 'number_of_moves' 
heuristic3 = 'number_of_tiles'

opponent_strategy1 = 'random'
opponent_strategy2 = 'optimal'

# In order to change the moves of the opponent either choose 'random' to play against a random opponent,
# or 'optimal' to play against an opponent also making optimal moves
opponent_strategy = 'random'

# In order to change which heuristic to run, change this heuristic equal to heuristic1|heuristic2||heuristic3 
heuristic = heuristic3 

# In order to change the batch size, change this number, and if you want the 'full' tree change batch_size ='full'
batch_size = 20 # or 'full'

# If you want to print all the subtrees change print_output = True
print_output = False

# change how many games you want to play, Minimax can take over 10 minutes to run 1 games depending on the heurstic 
and the subloop of games matched for a particular state
games = 5

# Enter the code block to play the game
