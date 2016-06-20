"""

"""
import pdb
import numpy
import matplotlib.pyplot as plt

import TicTacToe

class mcsolver (object):
    def __init__(self, game):
        self.game = game
        self.states_list(self.game.board.state.size)
        self.get_Q_dict()
        self.epsilon = 1.0
        #funcType = type(self.game.agent_x.random_control)
        #self.game.agent_x.random_control = funcType(self.e_greedy_control, self.game.agent_x, mcsolver)
        self.game.agent_x.random_control = self.e_greedy_control
        self.s_a_series = []



    def update_Q_dict(self, s_a_series, reward):
        """
        Updates the Q_dict give a series of actions and states and the total reward.
        s_a_series format: [s_a_1, s_a_2, . . . ]. s_a_1 = (state , action) in one n tuple.
        """
        for s_a in s_a_series:
            value = float(self.Q_dict[s_a][0])
            count = float(self.Q_dict[s_a][1])
            self.Q_dict[s_a][0] = (value*count + float(reward))/(count+1)
            self.Q_dict[s_a][1] = count + 1


    def e_greedy_control(self):
        """
        epsilon greedy control method
        controls agent's actions.
        defualt controller in the agent should be replaced by this function
        Epsilon
        """
        state = self.game.board.state.flatten().tolist()
        state = tuple(state)
        possible_acts = []
        for act in range(self.game.board.state.size):
            if self.Q_dict.has_key(state + (act,)):
                possible_acts.append(state+(act,))

        act = possible_acts[numpy.random.randint(len(possible_acts))]
        if numpy.random.rand() < self.epsilon:
            self.s_a_series.append(act)
            return act[-1]

        Max = self.Q_dict[act][0]
        best_act = act
        for act in possible_acts:
            if self.Q_dict[act][0] > Max:
                Max = self.Q_dict[act][0]
                best_act = act
        self.s_a_series.append(best_act)
        return best_act[-1]


    def states_list(self, size):
        """
        return a conplete list of all possible states
        """
        temp = [0]*size

        states = []
        if size == 1:
            return [[-1],[0],[1]]

        #3 possible values for each box
        for val in range(-1,2):
            states_ = self.states_list(size-1)
            for state in states_:
                states.append([val]+state)
        self.states = states
        return states


    def get_Q_dict(self):
        """
        generating the Q list.
        adds all possible actions for each state
        """
        self.Q_dict = {}

        updated_states = []
        for state in self.states:
            #checking board correctness
            if sum(state) not in (0,-1): #only x starts
                continue

            #checking if not already won
            self.game.board.set_state(state)
            if self.game.check_win() is not None:
                continue

            updated_states.append(state)
            for i in range(len(state)):
                if state[i] == 0:
                    self.Q_dict[tuple(state+[i])] = [0,0]
        self.states = updated_states
        return self.Q_dict


    def train(self, epochs = 1000000):
        """
        training a MCsolver.
        """
        m = len(self.states)
        rewards = numpy.zeros(epochs)
        disp_freq = 10000
        decay = 0.9

        for epch in range(epochs):
            r = numpy.random.randint(m)
            self.game.board.set_state(self.states[r])
            #self.game.board.set_state([0,0,0,0,0,0,0,0,0])
            #starting_player = self.game.check_turn()
            result = self.game.run_main()
            reward = result #* starting_player
            rewards[epch] = reward
            self.update_Q_dict(self.s_a_series, reward)
            self.s_a_series = []
            if epch%disp_freq == 0 and epch >= disp_freq:
                self.epsilon = self.epsilon*decay
                print ("rewards at epoch %s is and epsiolon is %s:"%(epch,self.epsilon))
                print (rewards[epch-disp_freq:epch].mean())


            #plt.imshow(self.game.board.draw())
            #plt.show()
        pdb.set_trace()


