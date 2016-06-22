try:
    import dqlttt.TicTacToe as TicTacToe
    from dqlttt.DQLsolver import dqlsolver

except ImportError:
    import os
    parentpath = os.path.dirname(os.path.abspath(__file__))
    os.sys.path.insert(0, parentpath)
    import dqlttt.TicTacToe as TicTacToe
    from dqlttt.DQLsolver import dqlsolver

import argparse

def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="choose either test or train. if test, provide the path to the model(.json) and weights(.h5)")
    parser.add_argument('--keras-model', help="path to the keras model definition (.json)")
    parser.add_argument('--keras-weights', help="path to the weigths file (.h5)")

    args = parser.parse_args()
    game =TicTacToe.game()
    solver = dqlsolver(game)
    if args.mode == 'train':
        print ("training a Deep Q learning method for playing TicTacToe")
        solver.train()

    elif args.mode == 'test':
        game.agent_o.set_control_type("human")
        solver.test(args.keras_model, args.keras_weights)

if __name__ == '__main__':
        main()

