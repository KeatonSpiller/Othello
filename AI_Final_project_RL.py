import numpy as np
import random, math, copy
from csv import reader
import PySimpleGUI as sg
from scipy.special import expit
import matplotlib.pyplot as plt

def read_csv(file_path):
    data = []
    dist = [INPUT_UNITS, HIDDEN_UNITS+1, INPUT_UNITS, HIDDEN_UNITS+1]
    with open(file_path, 'r') as obj:
        csv_obj = reader(obj)
        k = 0
        acc = dist[k]
        w = []
        i = 0
        for row in csv_obj:
            if i == acc:
                data.append(w)
                w = []
                k += 1
                acc += dist[k]
            x = []
            for j in row:
                num = float(j)
                x.append(num)
            w.append(x)
            i += 1
        data.append(w)
    return data

def write_csv(file_path, data):
    csv_writer = open(file_path, 'w')
    for r in data:
        for d in r:
            line = ','.join([str(x) for x in d]) 
            csv_writer.write(line)
            csv_writer.write('\n')
    csv_writer.close()

def checked_int(num):
    """ Method to check if a string can be convert to an integer
            argument:
                num: a string need to convert to integer
            return: the int converted from the input string
                    -1 if the input string is invalid
    """
    try:
        return int(num)
    except ValueError:
        return -1

def check_win(state):
    """ Count number of discs of each player to decide who won the game
            args:
                state: numpy array of checked state
            return: the color code of the winner
                    1 if no one won
                    -1 if the game has not completed yet
    """
    my_next = find_actions(state, MY_COLOR)
    op_next = find_actions(state, OP_COLOR)
    if len(my_next) == 0 and len(op_next) == 0:
        my_count = np.count_nonzero(state == MY_COLOR)
        op_count = np.count_nonzero(state == OP_COLOR)
        if my_count == op_count:
            return BLANK_COLOR, my_count
        elif my_count > op_count:
            return MY_COLOR, my_count
        else:
            return OP_COLOR, my_count
    else:
        return -1, 0
            
def find_actions(state, current_color):
    """ Find all possible actions from the current state
        args: 
            state: numpy vector represents state
            current_color: color code of current player
        return: all possible actions as the locations of next moves
                each action is in range(65)
    """
    actions = []
    arr = state.reshape(8,8)
    next_color = MY_COLOR if current_color == OP_COLOR else OP_COLOR
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] == current_color:
                moves = possible_moves(arr, [i,j], next_color, BLANK_COLOR)
                for m in range(8):
                    if moves[m] > 0:
                        action = -1
                        match m:
                            case 0:
                                action = i*8 + j - moves[m] - 1
                            case 1:
                                action = (i - moves[m] - 1)*8 + j - moves[m] - 1
                            case 2:
                                action = (i - moves[m] - 1)*8 + j
                            case 3:
                                action = (i - moves[m] - 1)*8 + j + moves[m] + 1
                            case 4:
                                action = i*8 + j + moves[m] + 1
                            case 5:
                                action = (i + moves[m] + 1)*8 + j + moves[m] + 1
                            case 6:
                                action = (i + moves[m] + 1)*8 + j
                            case 7:
                                action = (i + moves[m] + 1)*8 + j - moves[m] - 1
                        if action not in actions:
                            actions.append(action)
    return actions

def possible_moves(state, current_loc, who_next, terminated_disc):
    """ Find from 8 directions for all possible moves from current location
        args: 
            state: numpy 8x8 matrix represents current state
            current_loc: (row, column) of current disc
            who_next: color code of the next turn player
        return: an array with indices from horizontal-left go by clockwise
                0: horizontal-left 
                1: diagonal-left-up
                2: vertical-up
                3: diagonal-right-up
                4: horizontal-right
                5: diagonal-right-down
                6: vertical-down
                7: diagonal-left-down
            the values of the array are the number of discs can be flipped
    """
    ret = [0]*8
    i = current_loc[0]
    j = current_loc[1]
    # Check to left
    count_horizon = 0
    cont_horizon = True
    count_up = 0
    cont_up = True
    count_down = 0
    cont_down = True
    for k in range(j):
        if state[i][j-k-1] == who_next and cont_horizon:
            count_horizon += 1
        else:
            cont_horizon = False
        if i-k > 0 and state[i-k-1][j-k-1] == who_next and cont_up:
            count_up += 1
        else:
            cont_up = False
        if i+k < 7 and state[i+k+1][j-k-1] == who_next and cont_down:
            count_down += 1
        else:
            cont_down = False
    if count_horizon < j and state[i][j-count_horizon-1] == terminated_disc:
        ret[0] = count_horizon
    if count_up < j and count_up < i and state[i-count_up-1][j-count_up-1] == terminated_disc:
        ret[1] = count_up
    if count_down < j and count_down < 7-i and state[i+count_down+1][j-count_down-1] == terminated_disc:
        ret[7] = count_down
    # Check to right
    count_horizon = 0
    cont_horizon = True
    count_up = 0
    cont_up = True
    count_down = 0
    cont_down = True
    for k in range(7-j):
        if state[i][j+k+1] == who_next and cont_horizon:
            count_horizon += 1
        else:
            cont_horizon = False
        if i-k > 0 and state[i-k-1][j+k+1] == who_next and cont_up:
            count_up += 1
        else:
            cont_up = False
        if i+k < 7 and state[i+k+1][j+k+1] == who_next and cont_down:
            count_down += 1
        else:
            cont_down = False
    if count_horizon < 7-j and state[i][j+count_horizon+1] == terminated_disc:
        ret[4] = count_horizon
    if count_up < 7-j and count_up < i and state[i-count_up-1][j+count_up+1] == terminated_disc:
        ret[3] = count_up
    if count_down < 7-j and count_down < 7-i and state[i+count_down+1][j+count_down+1] == terminated_disc:
        ret[5] = count_down
    # Check vertical up
    count_up = 0
    count_down = 0
    for k in range(i):
        if state[i-k-1][j] == who_next:
            count_up += 1
        else:
            break
    for k in range(7-i):
        if state[i+k+1][j] == who_next:
            count_down += 1
        else:
            break
    if count_up < i and state[i-count_up-1][j] == terminated_disc:
        ret[2] = count_up
    if count_down < 7-i and state[i+count_down+1][j] == terminated_disc:
        ret[6] = count_down 
    return np.array(ret)

def get_next_state(state, action, current_color):
    """ Get the next state by opponent's discs
            args: 
                state: the current state
                action: the location that the player want to put his disc for his turn
                current_color: the color code of the current player
            return: a numpy array of the next state
    """
    arr = state.reshape(8,8)
    i = action // 8
    j = action % 8
    next_color = MY_COLOR if current_color == OP_COLOR else OP_COLOR
    moves = possible_moves(arr, [i,j], next_color, current_color)
    for k in range(len(moves)):
        if moves[k] > 0:
            arr[i][j] = current_color
            for m in range(moves[k]):
                match k:
                    case 0:
                        arr[i][j-m-1] = current_color
                    case 1:
                        arr[i-m-1][j-m-1] = current_color
                    case 2:
                        arr[i-m-1][j] = current_color
                    case 3:
                        arr[i-m-1][j+m+1] = current_color
                    case 4:
                        arr[i][j+m+1] = current_color
                    case 5:
                        arr[i+m+1][j+m+1] = current_color
                    case 6:
                        arr[i+m+1][j] = current_color
                    case 7:
                        arr[i+m+1][j-m-1] = current_color
    return arr.flatten()

def my_selection(state, prob):
    """ Select an action to create a new state. The Q-networks is also updated in this function
            args: 
                state: the current state
                prob: the probability to choos a randomized action or not 
            return: a pair of action and next state
    """
    possible_actions = find_actions(state, MY_COLOR)
    if  len(possible_actions) == 0:
        # Check if ending game
        if len(find_actions(state, OP_COLOR)) == 0:
            winner, count = check_win(state)
            return END_GAME, state
        else:
            return -1, state
    else:
        action = 0
        # Random a number
        n = random.randrange(100)/100
        if n < prob:
            # Random next action
            action = random.choice(possible_actions)
        else:
            # Choose action from Q networks
            h, o = forward_propagation(state, w2_ih, w2_ho)
            if is_debug:
                print(f'State: {state.reshape(8,8)}')
                print(f'Probability: {o}')
            ps = []
            for a in possible_actions:
                ps.append(o[a])
            max_value = max(ps)
            max_actions = []
            for a in possible_actions:
                if o[a] == max_value:
                    max_actions.append(a)
            action = random.choice(max_actions)    
        next_state = get_next_state(state, action, MY_COLOR)

        winner, count = check_win(next_state)
        # Update Q networks
        delta_hid, delta_out, hid2 = compute_error_terms(state, possible_actions, action, next_state, winner)
        update_weight(hid2, delta_out, ETA, 0.1, 2)
        tmp = np.copy(state)
        tmp = np.insert(tmp, 0, 1)
        update_weight(tmp, delta_hid, ETA, 0.1, 1)
        if winner != -1:
            return END_GAME, next_state   # Return ending game
        else:
            return action, next_state

def op_selection(state, action, pre_state, pre_action):
    """ Update Q-networks if oppenent won
            args:
                state: the current state
                action: the action of the opponent
                pre_state: the previous state of this state
                pre_action: the action leading to this state
            return: a pair of action and next state
    """
    next_state = get_next_state(state, action, OP_COLOR)
    winner, count = check_win(next_state)
    if winner != -1: # winning
        # Update Q networks
        possible_actions = find_actions(pre_state, MY_COLOR)
        delta_hid, delta_out, hid2 = compute_error_terms(pre_state, possible_actions, pre_action, next_state, winner)
        hid2.insert(0,1)
        update_weight(hid2, delta_out, ETA, 0.1, 2)
        tmp = np.copy(state)
        tmp = np.insert(tmp, 0, 1)
        update_weight(tmp, delta_hid, ETA, 0.1, 1)
        return END_GAME, next_state
    return action, next_state

def init_state():
    """ Create an initial state for the game
    """
    state = np.zeros((64,), dtype=int)
    state[27] = MY_COLOR
    state[28] = OP_COLOR
    state[35] = OP_COLOR
    state[36] = MY_COLOR
    return state

def training(who_first, prob):
    """ Training one episode. The program will take action based on Q learning algorithm.
        The opponent will take a random action
            args:
                who_first: the color code of the first player
                prob: the probability to get a random action.        
    """
    turn = who_first
    state = init_state()
    pre_state = init_state()
    pre_action = 0
    while True:
        if turn == MY_COLOR:
            pre_state = np.copy(state)
            pre_action, state = my_selection(state, prob)
            if pre_action == END_GAME:
                return check_win(state)
            turn = OP_COLOR
            
        else:
            possible_actions = find_actions(state, OP_COLOR)
            if len(possible_actions) == 0:
                turn = MY_COLOR
                continue
            action = random.choice(possible_actions)
            action, state = op_selection(state, action, pre_state, pre_action)
            if action == END_GAME:
                return check_win(state)
            turn = MY_COLOR

def bulk_training():
    global w1_ih, w1_ho, w2_ih, w2_ho
    for n in range(int(EPSILON//DELTA)):
        prob = EPSILON - n*(DELTA)
        print(f'Training group {n+1}')
        for k in range(M_EPOCHS):
            # Start epoch
            for i in range(NUM_EPISODE):
                turn = OP_COLOR if i%2 == 0 else MY_COLOR
                training(turn, prob)
            w1_ih = copy.deepcopy(w2_ih)
            w1_ho = copy.deepcopy(w2_ho)

def render_state(window, state):
    """ Render the board game on UI by a state
            args: 
                window: the window object of the UI
                state: the current state that need to render
    """
    for k in range(64):
        if state[k] != 0:
            i = k // 8
            j = k % 8
            window[(i,j)].update(my_char)
            if state[k] == OP_COLOR:
                window[(i,j)].update(button_color=('White','Green'))
            else:
                window[(i,j)].update(button_color=('Black','Green'))
            

def play_game(who_first):
    """ UI for playing the game
            args:
                who_first: the color code of the player who goes fisrt
    """
    layout = [
        ([sg.B(' ',size=(3,1), font='Arial 30', button_color=('Black','Green'), key=(j,i)) for i in range(8)] for j in range(8)),
        [sg.T('Player X', size=(20,4), font='Arial 16', key='player')]
        ]
    window = sg.Window('Tic Skeleton', layout, finalize=True)
    state = init_state()
    pre_state = init_state()
    pre_action = 0
    is_playing = True
    if who_first == MY_COLOR:
        pre_action, state = my_selection(state, 0)
        window['player'].update('Program goes first.')
    else:
        window['player'].update('You go first.')
    render_state(window, state)
    while True:             # Event Loop
        event, values = window.read()
        if event in (None, 'Exit'):
            break
        
        current_marker = window[event].get_text()
        if is_playing:
            if current_marker == ' ':
                # Human's turn
                action = 8 * event[0] + event[1]
                possible_actions = find_actions(state, OP_COLOR)
                if len(possible_actions) > 0:
                    if action not in possible_actions:
                        continue
                    action, state = op_selection(state, action, pre_state, pre_action)
                    render_state(window, state)
                    if action == END_GAME:
                        winner, count = check_win(state)
                        if winner == BLANK_COLOR:
                            window['player'].update('No one won!')
                        elif winner == MY_COLOR:
                            window['player'].update('The program won!')
                        else:
                            window['player'].update('You won!')
                        is_playing = False
                        continue
                # Program's turn
                pre_state = np.copy(state)
                pre_action, state = my_selection(state, 0)
                render_state(window, state)
                i = pre_action // 8
                j = pre_action % 8
                key = (i,j)
                if pre_action == END_GAME:
                    winner, count = check_win(state)
                    if winner == BLANK_COLOR:
                        window['player'].update('No one won!')
                    elif winner == MY_COLOR:
                        window['player'].update('The program won!')
                    else:
                        window['player'].update('You won!')
                    is_playing = False
                    continue
    window.close()


def self_playing(weights, who_first):
    """ Self playing game based on the current and previous weights of the neuron networks
            args:
                weights: the weights matrix of NN
                who_first: the color code of the player who goes first
            return: color code of the winner
    """
    turn = who_first
    state = init_state()
    pre_state = init_state()
    pre_action = 0
    while True:
        if turn == MY_COLOR:
            pre_state = np.copy(state)
            pre_action, state = my_selection(state, 0)
            if pre_action == END_GAME:
                return check_win(state)
            turn = OP_COLOR
        else:
            possible_actions = find_actions(state, OP_COLOR)
            if len(possible_actions) == 0:
                turn = MY_COLOR
                continue
            h, o = forward_propagation(state, weights[2], weights[3])
            ps = []
            for a in possible_actions:
                ps.append(o[a])
            max_value = max(ps)
            max_actions = []
            for a in possible_actions:
                if o[a] == max_value:
                    max_actions.append(a)
            action = random.choice(max_actions)   
            action, state = op_selection(state, action, pre_state, pre_action)
            #print(state.reshape(8,8))
            if action == END_GAME:
                return check_win(state)
            turn = MY_COLOR

def init_weight(x,y):
    """ Initialize a weight matrix
            args:
                x, y: row, column size of the weight matrix
            return: the x*y matrix
    """
    ret = []  
    for i in range(x):
        wi = []
        for j in range(y):
            if j % 2 == 0:
                wi.append(0.1)
            else:
                wi.append(-0.1)
        ret.append(wi)
    return ret

def init_delta_weight(x,y):
    """ Initialize a zero weight matrix
            args:
                x, y: row, column size of the weight matrix
            return: the x*y matrix
    """
    ret = []
    for i in range(x):
        wi = [0.0 for j in range(y)]
        ret.append(wi)
    return ret

def forward_propagation(state, w1, w2):
    """ Method to calculate the hidden and output nodes by one set of input
        arguments:
            data: a vector contains 785 input values
            w1: the weight matrix of input and hidden nodes
            w2: the weight matrix of hidden and output nodes
        return: hidden nodes, output nodes
    """
    state = np.array(state)
    state = np.insert(state, 0, 1)  # add bias input
    sh = np.dot(state, w1)
    h = expit(sh)           # apply sigmod function
    h = np.insert(h, 0, 1)
    so = np.dot(h, w2)
    o = expit(so)
    return list(h), list(o)

def compute_error_terms(state, possible_actions, action, next_state, game_result):
    """ Calculate the delta values of input-hidden and hidden-output for a single action
            args:
                state: the input data
                lable: a number 0-9 of the input data
            return: delta_k, delta_j, and hidden values
    """
    global w1_ih, w1_ho, w2_ih, w2_ho
    #Forward propagation of predicted networks
    hid2, out2 = forward_propagation(state, w2_ih, w2_ho)
    delta_out = []
    # Find error terms of hidden-output
    for a in range(len(out2)):
        t = 0.0
        if a in possible_actions:
            if game_result != -1:
                if game_result == MY_COLOR:
                    t = 0.9
                elif game_result == OP_COLOR:
                    t == 0.1
                else:
                    t = 0.5
            else:
                #Forward propagation of target networks
                hid1, out1 = forward_propagation(next_state, w1_ih, w1_ho)
                arr = []
                for k in possible_actions:
                    arr.append(out1[k])
                t = out1[a] + ETA * (GAMMA * max(arr) - out1[a]) # target value
        else:
            t = 0.5
        y = out2[a]
        delta_out.append(y * (1 - y) * (t - y))
    # Find error terms of input-hidden
    delta_hid = []
    for j in range(len(hid2)-1):
        s = np.dot(w2_ho[j], delta_out)
        d = hid2[j+1]*(1-hid2[j+1])*s
        delta_hid.append(d)
        
    return delta_hid, delta_out, hid2

def update_weight(input_vector, delta, rate, momen, op):
    """ Update weights based on model's parameters
            args: 
                input_vector: input or hidden data
                delta: a delta vector corresponding to the weight
                rate: the learning rate
                momen: the momentum
                op: 1 if input-hidden or 2 if hidden-output
    """
    global w2_ih, w2_ho, past_w2_ih, past_w2_ho
    w = []
    past_w = []
    if op == 1:
        w = w2_ih
        past_w = past_w2_ih
    else:
        w = w2_ho
        past_w = past_w2_ho
    delta_w = []
    for j in range(len(w)):
        pre = rate * input_vector[j]
        d = map(lambda x: x*pre, delta)
        delta_w.append(list(d))
    alpha = np.multiply(past_w, momen)
    delta_w = np.add(delta_w, alpha)
    arr = np.add(w, delta_w)
    if op == 1:
        w2_ih = copy.deepcopy(arr)
        past_w2_ih = copy.deepcopy(delta_w)
    else:
        w2_ho = copy.deepcopy(arr)
        past_w2_ho = copy.deepcopy(delta_w)

""" Main Method from Here
"""
# Environment paramenters
BLANK_COLOR = 0
MY_COLOR = 1
OP_COLOR = 2
END_GAME = -2
my_char = chr(11044)
op_char = chr(9711)
EPSILON = 0.1
DELTA = 0.02
M_EPOCHS = 5
NUM_EPISODE = 100
ETA = 0.4
GAMMA = 0.2
FILE_TARGET_IH = 'q_target_wih.csv'
FILE_TARGET_HO = 'q_target_who.csv'
FILE_PREDICT_IH = 'q_predict_wih.csv'
FILE_PREDICT_HO = 'q_predict_who.csv'
FILE_WEIGHT = 'q_weights.csv'
INPUT_UNITS = 65
OUTPUT_UNITS = 64
HIDDEN_UNITS = 16
is_debug = False
#Initialize weight matrix
w1_ih = []
w1_ho = []
w2_ih = []
w2_ho = []
past_w2_ih = init_delta_weight(INPUT_UNITS, HIDDEN_UNITS)
past_w2_ho = init_delta_weight(HIDDEN_UNITS + 1, OUTPUT_UNITS)
weights = []

while True:
    print('PROGRAM FEATURES:')
    print('1. Train the program from scratch')
    print('2. Continue training the program')
    print('3. Play the game')
    print('4. Self playing')
    print('5. Plot scores in 100 games')
    print('6. Find meand and sd of 1000 games')
    num_op = 0
    is_debug = False
    while True:
        num_in = input('Enter 1, 2, 3, 4, 5, 6: ')
        num_op = checked_int(num_in)
        if num_op in [1,2,3,4,5,6]:
            break
        else:
            print('Error: input option must be 1, 2, 3, 4, 5, 6')
    if num_op == 1:
        #Initialize weight matrix
        w1_ih = init_weight(INPUT_UNITS, HIDDEN_UNITS)          # Target Weights input-hidden
        w1_ho = init_weight(HIDDEN_UNITS + 1, OUTPUT_UNITS)     # Target Weights hidden-output
        w2_ih = copy.deepcopy(w1_ih)                            # Predict Weights input-hidden
        w2_ho = copy.deepcopy(w1_ho)                            # Predict Weights hidden-output
        weights = [w1_ih, w1_ho, w2_ih, w2_ho]
    else:
        weights = read_csv(FILE_WEIGHT)
        w1_ih = weights[0]
        w1_ho = weights[1]
        w2_ih = weights[2]
        w2_ho = weights[3]
    if num_op == 1 or num_op == 2:
        bulk_training()

    elif num_op == 3:
        #is_debug = True
        is_your = input('Would you like to go first? (y/n): ')
        if is_your == 'y':
            play_game(OP_COLOR)
        else:
            play_game(MY_COLOR)
    elif num_op == 4:
        qfile = input('File name of weights: ')
        weights = read_csv(qfile)

        num_win = 0
        num_lose = 0
        for i in range(NUM_EPISODE):
            turn = OP_COLOR if i%2 == 0 else MY_COLOR
            result, count = self_playing(weights, turn)
            if result == MY_COLOR:
                num_win += 1
            elif result == OP_COLOR:
                num_lose +=1
        print(f'Percentage winning: {num_win*100/NUM_EPISODE}%')
        print(f'Percentage losing: {num_lose*100/NUM_EPISODE}%')
    elif num_op == 5:

        y1 = []
        y2 = []
        x = []
        for k in range(100):
            weights = read_csv('weights_v1.csv')
            result11, count11 = training(MY_COLOR, 0)
            result12, count12 = training(OP_COLOR, 0)
            count1 = (count11 + count12)/2
            result21, count21 = self_playing(weights, MY_COLOR)
            result22, count22 = self_playing(weights, OP_COLOR)
            count2 = (count21 + count22)/2
            x.append(k+1)
            y1.append(count1)
            y2.append(count2)
        plt.xlabel(f"Games")
        plt.ylabel("Scores")
        plt.title("Scores in 100 Games")
        plt.plot(x,y1, label="Random player")
        plt.plot(x,y2, label="Self-playing")
        plt.legend(loc="lower right")
        plt.show()
    else:
        num_win = 0
        num_count = []
        #weights = read_csv('weights_v1.csv')
        for i in range(1000):
            turn = OP_COLOR if i%2 == 0 else MY_COLOR
            result, count = training(turn, 0)
            #result, count = self_playing(weights, turn)
            num_count.append(count)
            if result == MY_COLOR:
                num_win += 1
        num_count = np.array(num_count)
        print(f'Win: {num_win}')
        print(f'Mean: {np.mean(num_count)}')
        print(f'sd: {np.std(num_count)}')

    write_csv(FILE_WEIGHT, weights)
    con_in = input('Do you want to continue? (y/n): ')
    if con_in == 'y':
        continue
    else: 
        break
