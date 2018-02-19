from board import Board
from players import *

class Game:
    def __init__(self, player1, player2):
        self.board = Board()
        self.players = [player1, player2]

    def run_training(self):
        '''
        runs a single game and returns the winner, i.e. 0 (stale), 1, or 2

        returns the winner
        '''
        # prepare a new board
        self.board = Board()

        # reset players for each round
        [player.reset() for player in self.players]

        player_index = 0

        while not self.is_over():
            player = self.players[player_index]

            position = player.get_next_move(self.board)
            self.board.place(player.get_token(), position)

            player_index = (player_index + 1) % 2

        # inform the players about the outocome of the game
        winner = self.board.find_winner()
        [player.learn(winner) for player in self.players]

        return winner

    def train_players(self, num_epochs=100):
        '''
        runs the single training game a number of times
        '''
        # make sure players will actually learn
        [player.set_learning(True) for player in self.players]

        winners = []

        for i in range(num_epochs):
            if i % 5000 == 0: print(i)

            winner = self.run_training()
            winners.append(winner)

        # deactivate learning after training
        [player.set_learning(False) for player in self.players]

        return winners

    def play(self):
        '''
        runs a single game and displays the board on each step
        '''
        player_index = 0

        while not self.is_over():
            self.board.display()

            player = self.players[player_index]

            position = player.get_next_move(self.board)
            self.board.place(player.get_token(), position)

            player_index = (player_index + 1) % 2

        winner = self.board.find_winner()

        if winner == 0:
            print('Nobody won')
        else:
            print(f'Player {winner} won!')

    def is_over(self):
        '''
        returns true if there is a winner, else false
        '''
        winner = self.board.find_winner()

        return winner is not None
