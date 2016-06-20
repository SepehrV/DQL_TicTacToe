"""
Deep Q learning implementation for graphical TicTacToe game
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop

from keras.utils import np_utils

import numpy
import time
import pdb

import TicTacToe


class dqlsolver(object):
    def __init__ (self, game):
        """
        Initializing the deep Q learning solver.
        """
        self.game = game
        self.action_space = self.game.board.state.size
        self.current_Q_vals = [0]*self.action_space
        self.epsilon = 1.0
        self.gamma = 0.9
        self.game.agent_x.random_control = self.e_greedy_control
        self.model = self.init_model()


    def init_model(self):
        """
        initilaizes the keras CNN networks.
        """
        img_rows = self.game.board.draw().shape[0]
        img_cols = self.game.board.draw().shape[1]

        model = Sequential()
        model.add(Convolution2D(32, 3, 3,
                                border_mode='valid',
                                input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_space))
        model.add(Activation('linear'))
        model.compile(loss="mse", optimizer=RMSprop())
        return model


    def update_model(self, act, reward, is_terminal=False):
        y = self.current_Q_vals

        if is_terminal:
            y[0,act] = reward
        else:
            #setting the state to the state after the action
            self.game.board.change_state(self.game.agent_x, act)

            #getting the image after the move
            img = self.game.board.draw()
            Q_vals_next = self.model.predict(img.reshape(1,1,img.shape[0], img.shape[1]), batch_size =1)
            y[0,act] = reward + self.gamma*Q_vals_next.max()

            #resetting to actual state
            self.game.board.change_state(self.game.agent_x, act, reset = True)


        self.model.fit(self.current_img.reshape(1, 1, self.current_img.shape[0], self.current_img.shape[1]),
                y,
                batch_size=1,
                nb_epoch=1,
                verbose=0)


    def e_greedy_control(self):
        """
        epsilon greedy control method
        controls agent's actions.
        defualt controller in the agent should be replaced by this function
        Epsilon
        """

        img = self.game.board.draw()
        self.current_img = img

        Q_vals = self.model.predict(img.reshape(1,1,img.shape[0], img.shape[1]), batch_size =1)
        self.current_Q_vals = Q_vals

        if numpy.random.rand() < self.epsilon:
            act = numpy.random.randint(self.action_space)
        else:
            temp = Q_vals.flatten()
            max_q = numpy.where(temp == temp.max())
            act = int(max_q[0][0])

        #checking if the move is allowed or not.
        #if not, since it is a terminal move, we leave it to get handled later
        play = (act//self.game.board.shape[1], act%self.game.board.shape[1])
        if self.game.board.state[play] == 0:
            self.update_model(act, 0, is_terminal = False)

        self.last_act = act

        return act


    def train(self, epochs = 1000000):
        """
        training a DQLsolver.
        """
        rewards = numpy.zeros(epochs)
        disp_freq = 1000
        decay = 0.95

        print ("start of training")
        start_time = time.time()

        for epch in range(epochs):
            #pdb.set_trace()
            self.game.board.set_state([0,0,0,0,0,0,0,0,0])

            result = self.game.run_main()
            reward = result #* starting_player
            self.update_model(self.last_act, reward, is_terminal = True)
            rewards[epch] = reward

            if epch%disp_freq == 0 and epch >= disp_freq:
                if self.epsilon*decay < 0.1:
                    decay = 1.0
                self.epsilon = self.epsilon*decay
                print ("rewards at epoch %s is and epsiolon is %s:"%(epch, self.epsilon))
                print (rewards[epch-disp_freq:epch].mean())
                print ("%s epochs are done in %s"%(epch, time.time()-start_time))



