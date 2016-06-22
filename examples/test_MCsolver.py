try:
    import dqlttt.TicTacToe as TicTacToe
    from dqlttt.MCsolver import mcsolver

except ImportError:
    import os
    parentpath = os.path.dirname(os.path.abspath(__file__))
    os.sys.path.insert(0, parentpath)
    import dqlttt.TicTacToe as TicTacToe
    from dqlttt.MCsolver import mcsolver


def main():
    game =TicTacToe.game()
    solver = mcsolver(game)
    print ("training a MonteCarlo base Q learning method.")
    solver.train()


if __name__ == '__main__':
        main()

