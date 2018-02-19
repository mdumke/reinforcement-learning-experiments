import numpy as np
import hashlib

class Board:
    '''
    Tic-tac-toe board for two players. Player 1 uses token 1, player 2
    uses -1.
    '''
    def __init__(self):
        self.state = np.zeros((3, 3)).astype('int')

    def place(self, token, position):
        '''
        sets the token at the specified position

        NOTE: Mutates self.state

        @param token: one of 1, -1
        @param position: tuple of integers with (0, 0) <= (i, j) < (3, 3)
        '''
        assert token in [-1, 1], f'invalid token {token}'
        assert type(position) == type((0, 0)), 'position must be a tuple'

        if self.state[position] != 0:
            assert False, 'position is taken'

        self.state[position] = token

    def find_winner(self):
        '''
        returns the winner (1 or 2) if one can be found or None
        '''
        # check rows
        if self.state.sum(axis=1).max() == 3: return 1
        if self.state.sum(axis=1).min() == -3: return 2

        # check columns
        if self.state.sum(axis=0).max() == 3: return 1
        if self.state.sum(axis=0).min() == -3: return 2

        # check diagonals
        if self.state.trace() == 3 or np.fliplr(self.state).trace() == 3: return 1
        if self.state.trace() == -3 or np.fliplr(self.state).trace() == -3: return 2

        # check stalemate
        if self.is_full(): return 0

        return None

    def hash(self):
        '''
        returns the board state as unique hash
        '''
        return hashlib.sha256(self.state).hexdigest()

    def is_full(self):
        '''
        returns true if there are not positions left
        '''
        return np.all(self.state)

    def get_free_positions(self):
        '''
        returns a numpy-array of tuples representing free board positions
        '''
        free = []

        for i in range(3):
            for j in range(3):
                token = self.state[i][j]

                if token == 0:
                    free.append((i, j))

        return np.array(free, dtype=[('row', np.uint8), ('col', np.uint8)])

    def display(self):
        '''
        prints a representation of the current board-state
        '''
        output = '\n'

        for i in range(3):
            for j in range(3):
                token = self.state[i][j]
                if token != -1:
                    output += ' '

                output += str(token)

            output += '\n'

        print(output)
