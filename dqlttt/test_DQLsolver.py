from DQLsolver import dqlsolver
import TicTacToe
import pdb
import numpy

game =TicTacToe.game()
solver = dqlsolver(game)
solver.train()
#solver.test()

