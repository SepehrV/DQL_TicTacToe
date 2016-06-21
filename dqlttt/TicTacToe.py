"""
TicTacToe game:
    Define the board shape in the board class.
    Define win conditions in the game class.
    Add controller function inside agent for your own control method.
"""
import numpy
import pdb
import Image, ImageDraw
import matplotlib.pyplot as plt
import logging

_logger = logging.getLogger(__name__)

class game(object):
    """
    main class of the tic tac toe game. containing board and agents.
    """
    def __init__(self):
        """
        Creating the board and agents.
        """
        self.win_rule = 3 # how many marks in a row to be considered as a point
        self.win_cond = 1 #how many point to win
        self.board = board()
        self.agent_x = agent('x', self.board)
        self.agent_o = agent('o', self.board)
        self.create_filters()


    def run_main(self):
        while self.board.get_allowed_moves().sum() > 0:
            try:
                if self.check_turn() == 1:
                    self.board.change_state(self.agent_x, self.agent_x.move())

                elif self.check_turn() == -1:
                    self.board.change_state(self.agent_o, self.agent_o.move())

            except ValueError:
                ## wrong move command
                return -1

            if self.check_win() == 1:
                #print ("x wins")
                #plt.imshow(self.board.draw())
                #plt.show()
                return 1

            if self.check_win() == -1:
                #print ("o wins")
                #plt.imshow(self.board.draw())
                #plt.show()
                return -1

            if self.check_win() == 0:
                pdb.set_trace()
                #print ("tie")
                #plt.imshow(self.board.draw())
                #plt.show()
                return 0
        return 0


    def check_turn(self):
        """
        returns -1 if it is o's turn.
        returns 1 if it is x's turn.
        returns 0 if the board is wrong
        x starts by default
        """
        turn = self.board.state.sum()
        if turn in (0,-1):
            return 1
        if turn == 1:
            return -1
        return 0


    def create_filters(self):
        """
        create filter to check win
        """
        filters = []
        f_temp = numpy.zeros((self.win_rule, self.win_rule))
        for row in range(f_temp.shape[0]):
            t = f_temp.copy()
            t[row,:] = 1
            filters.append(t)

        for col in range(f_temp.shape[1]):
            t = f_temp.copy()
            t[:,col] = 1
            filters.append(t)
        filters.append(numpy.eye(f_temp.shape[0]))
        filters.append(numpy.fliplr(numpy.eye(f_temp.shape[0])))
        self.filters = filters


    def check_win(self):
        """
        checking win condition:
        0 if tie
        1 if x wins
        -1 if o wins
        """
        m = self.board.shape[0]
        n = self.board.shape[1]
        points_x = 0
        points_o = 0
        for i in range(m-(m//2+1)):
            for j in range(n-(n//2+1)):
                for f in self.filters:
                    if (self.board.state[i:m,i:n]*f).sum() == self.win_rule:
                        points_x = points_x + 1

                    if (self.board.state[i:m,i:n]*f).sum() == -self.win_rule:
                        points_o = points_o + 1
        if points_x == points_o and self.board.get_allowed_moves().sum() == 0:
            return 0
        if points_x >= self.win_cond:
            return 1
        if points_o >= self.win_cond:
            return -1


class agent(object):
    """
    agent class.
    making moves.
    """
    def __init__(self, name, board):
        """
        Init an agent with name (x or o)
        """
        assert (name in ('x', 'o')),"name for an agent is not acceptatble!"
        self.name = name
        self.board = board


    def move(self, control='random'):
        """
        makes a move based on given control
        """
        if control == 'human':
            return self.human_control()
        if control == 'random':
            return self.random_control()


    def human_control(self):
        """
        gets two input from human for row and column of the square in the board
        """
        ("select a square from (0,0) to %s \n"%(board.state.shape))
        row = input("enter row\n")
        col = input("enter col\n")
        return (row,col)


    def random_control(self):
        """
        return a random allowed move
        """
        if self.board.get_allowed_moves().sum() == 0:
            raise ValueError("move requested with no space left")

        allowed = self.board.get_allowed_moves().flatten()
        sq =numpy.random.randint(allowed.sum()) + 1
        choose = numpy.zeros(allowed.size)
        for i in range(allowed.size):
            if allowed[i]:
                sq = sq -1
            if sq ==0:
                break
        choose[i] = 1

        return numpy.where( choose.reshape(self.board.shape) == 1)


    def intelligent_control(self, state):
        """
        place holder for an intelligent control to takes its place
        """
        print ("no overriden function yet!")
        print ("replace this function with a smart one")
        raise ValueError("intelligent_control called with no overriden function")

class board(object):
    """
    board class.
    containg state.
    drawing of the board.
    """

    def __init__(self, shape=(3,3), image_scale = 10):
        """
        initializing the board size.
        """
        self.state = numpy.zeros(shape)
        self.shape = shape
        self.image_scale = 10


    def draw(self):
        """
        draws the board state.
        size of the output image is defined by image_scale*shape
        """
        image = Image.new("RGB", (self.image_scale*self.shape[0], self.image_scale*self.shape[1]))
        D = ImageDraw.Draw(image)

        for i in range(1,self.shape[0]):
            D.line([i*self.image_scale,0,i*self.image_scale, self.shape[1]*self.image_scale])

        for j in range(1,self.shape[0]):
            D.line([0,j*self.image_scale,self.shape[0]*self.image_scale,j*self.image_scale])

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                pos = (int(j*self.image_scale + 2), int(i*self.image_scale))
                if self.state[i,j] == 1:
                    D.text(pos, 'x')
                elif self.state[i,j] == -1:
                    D.text(pos, 'o')
        del D
        np_img = numpy.asarray(image, dtype="float32")

        return np_img[:,:,0]/float(255)


    def change_state(self, agent, play, reset=False):
        """
        performs a move for agent given by play.
        """
        if type(play) is int:
            play = (play//self.shape[1], play%self.shape[1])

        if reset:
            self.state[play] = 0
            return 1

        if self.state[play] != 0:
            raise ValueError("place is taken!")

        if agent.name == 'x':
           self.state[play] = 1
        else:
           self.state[play] = -1
        return 1


    def __str__(self):
        """
        string representation of the board. prints the board
        """
        return str(self.state)


    def get_allowed_moves(self):
        """
        return all allowed moves
        """
        return self.state == 0


    def set_state(self, state):
        """
        set the current state from outside.
        good for initializing the state with a random one
        """
        state = numpy.array(state)
        if state.shape != self.shape:
            state = state.reshape(self.shape)
        self.state = state


