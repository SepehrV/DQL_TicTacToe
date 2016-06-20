from MCsolver import mcsolver
import TicTacToe
import pdb
import numpy

#pdb.set_trace()
game =TicTacToe.game()
solver = mcsolver(game)
solver.train()


