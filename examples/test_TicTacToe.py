"""
An example of TicTacToe game where both agents are conotrlled by humans
"""
try:
    import dqlttt.TicTacToe as TicTacToe

except ImportError:
    import os
    parentpath = os.path.dirname(os.path.abspath(__file__))
    os.sys.path.insert(0, parentpath)
    import dqlttt.TicTacToe as TicTacToe

game = TicTacToe.game()
game.agent_x.set_control_type('human')
game.agent_o.set_control_type('human')
game.run_main(show=True)
