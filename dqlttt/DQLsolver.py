"""
Deep Q learning implementation for graphical TicTacToe game
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.models import model_from_json

from keras.utils import np_utils

import numpy
import time
import random
import matplotlib.pyplot as plt

import TicTacToe


class dqlsolver(object):
    def __init__ (self, game):
        """
        Initializing the deep Q learning solver.
        """
        self.game = game
        self.x_shape = self.game.board.draw().shape
        self.action_space = self.game.board.state.size
        self.current_Q_vals = [0]*self.action_space
        self.epsilon = 1.0
        self.gamma = 0.9
        self.game.agent_x.random_control = self.e_greedy_control
        self.model = self.init_model()

        self.exp_count = 10000
        self.memory = []
        #self.exp_imgs = numpy.zeros((self.exp_count, self.x_shape[0], self.x_shape[1]))
        #self.exp_labels = numpy.zeros((self.exp_count, self.action_space))


    def init_model(self):
        """
        initilaizes the keras CNN networks.
        the model is very simple:
        CNN->RELU->CNN->RELU->POOL->FC->FC
        it is being used for computing Q for all actions in a given state
            simultanously.
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


    def load_model(self, model_path, weight_path):
        """
        loading a keras model and its weights
        """
        self.model = model_from_json(open(model_path).read())
        self.model.load_weights(weight_path)



    def update_model(self, act, reward, is_terminal=False):
        """
        updates the model using belman equation.
        act is being used to find the best Q for the next step and therefore computing the E(Q).
        reward is +1 for win -1 for lose 0 for tie and -punish for making illegal move.
            it is defined inside TicTacToe.
        is_terminal flag indicates final move (win,lose,tie,illegal move)
        """
        y = self.current_Q_vals
        next_img = None

        if is_terminal:
            y[0,act] = reward
        else:
            #setting the state to the state after the action
            self.game.board.change_state(self.game.agent_x, act)

            #getting the image after the move
            next_img = self.game.board.draw()
            Q_vals_next = self.model.predict(next_img.reshape(1,1,next_img.shape[0], next_img.shape[1]), batch_size =1)
            y[0,act] = reward + self.gamma*Q_vals_next.max()

            #resetting to actual state
            self.game.board.change_state(self.game.agent_x, act, reset = True)

        #storing transitions for experinece replay
        self.memory.append([self.current_img, act, reward, next_img])

        #self.model.fit(self.current_img.reshape(1, 1, self.current_img.shape[0], self.current_img.shape[1]),
        #        y,
        #        batch_size=1,
        #        nb_epoch=1,
        #        verbose=0)


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


    def train(self, epochs = 100000):
        """
        training a DQLsolver.
        """
        f = open("log", 'w')
        rewards = numpy.zeros(epochs)
        # works as epilon update rate too
        disp_freq = 2000
        decay = 0.95
        # does one replay train each replay_freq episodes
        replay_freq = 1

        print ("start of training")
        start_time = time.time()
        mini_batch_size = 32

        for epch in range(epochs):
            self.game.board.set_state([0,0,0,0,0,0,0,0,0])

            result = self.game.run_main()
            reward = result #* starting_player
            self.update_model(self.last_act, reward, is_terminal = True)
            rewards[epch] = reward

            #experinece replay pool
            if len(self.memory) > self.exp_count + replay_freq:
                mini_batch = random.sample(self.memory, mini_batch_size)
                X_train = []
                Y_train = []
                for exp in mini_batch:
                    State, Action, Reward, New_state = exp
                    Q = self.model.predict(State.reshape(1,1,self.x_shape[0],self.x_shape[1]), batch_size=1)
                    y = Q.copy()

                    if New_state is not None: # non-terminal state
                        New_Q = self.model.predict(New_state.reshape(1,1,self.x_shape[0],self.x_shape[1]), batch_size=1)
                        maxQ = New_Q.max()
                        update = Reward + self.gamma*maxQ
                    else:
                        update = Reward

                    y[0][Action] = update
                    X_train.append(State.reshape(1,self.x_shape[0],self.x_shape[1]))
                    Y_train.append(y.flatten())

                #training the network on a mini batch from memory
                self.model.fit(numpy.array(X_train),
                        numpy.array(Y_train),
                        batch_size=mini_batch_size,
                        nb_epoch=1,
                        verbose=0)

                #removing oldest elem in the memory
                del self.memory[0:replay_freq]

            # displaying the progress
            if epch%disp_freq == 0 and epch >= disp_freq:
                if self.epsilon*decay < 0.1:
                    decay = 1.0
                self.epsilon = self.epsilon*decay
                print ("rewards at epoch %s is %s and epsiolon is %s:"%(epch,rewards[epch-disp_freq:epch].mean(), self.epsilon))
                f.write("rewards at epoch %s is %s  and epsiolon is %s\n:"%(epch,rewards[epch-disp_freq:epch].mean(), self.epsilon))
                print ("%s epochs are done in %s"%(epch, time.time()-start_time))
        f.close()

        open("model",'w').write(self.model.to_json())
        self.model.save_weights('model_weights.h5')


    def test(self, model_path, weight_path):
        """
        test function.
        loads the model
        replaces e_greedy_control with test control
        runs test epochs.
        """
        #model_path = "model.json"
        #weight_path = "model_weights.h5"
        self.load_model(model_path, weight_path)
        self.game.agent_x.random_control = self.test_control

        test_epochs = 500
        rewards = numpy.zeros(test_epochs)
        for epch in range(test_epochs):
            self.game.board.set_state([0,0,0,0,0,0,0,0,0])

            result = self.game.run_main(show=True)
            reward = result #* starting_player
            rewards[epch] = reward

        print ("rewards after %s epochs is %s:"%(epch, rewards.mean()))


    def test_control(self):
        """
        agent controller during testing
        """

        img = self.game.board.draw()
        self.current_img = img

        Q_vals = self.model.predict(img.reshape(1,1,img.shape[0], img.shape[1]), batch_size =1)
        self.current_Q_vals = Q_vals

        temp = Q_vals.flatten()
        max_q = numpy.where(temp == temp.max())
        act = int(max_q[0][0])
        self.last_act = act

        return act









