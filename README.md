#Replicating Deep Q-Learning algorithm for the tic-tac-toe game.

This simple example of Deep Q leanrning is inspired by [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). It is meant to be basic so it would be easy to see and modify the main algorithm. 

The package includes: 
1. A Deep Q learning solver for playing graphical Tic-Tac-Toe (the agent only sees an image of the board and does not have access to the internal state of the game.)
2. A Monte Carlo based Q learning solver for playing Tic-Tac-Toe. (agent works on game's internal state space )
3. An implementation for Tic-Tac-Toe game. 

##Install
This package requires [keras](https://github.com/fchollet/keras).

To install:
```bash
git clone https://github.com/SepehrV/DQL_TicTacToe
cd DQL_TicTacToe
sudo python setup.py install
```

##Usage
To play against a trained model using Deep Q learning:
```bash
python examples/test_DQLsolver.py test --keras-model models/model.json --keras-weights models/model_weights.h5
```

To train your own Deep Q learning based agent:
```bash
python examples/test_DQLsolver.py train
```

To train a Monte Carlo base agent:
```bash
python examples/test_DQLsolver.py train
```


