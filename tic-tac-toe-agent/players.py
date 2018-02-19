import re
import numpy as np

class Player:
    '''
    general player class
    '''
    def __init__(self, player_id):
        '''
        initializes the manual player

        @param player_id specifies the player, either 1 ('x') or 2 ('o')
        '''
        assert player_id in [1, 2], 'token must be one of 1, 2'

        self.id = player_id
        self.token = 1 if player_id == 1 else -1

    def get_id(self):
        '''
        returns the player's id, i.e. 1 or 2
        '''
        return self.id

    def get_token(self):
        '''
        return the token this player is using
        '''
        return self.token

    def reset(self):
        '''
        prepares the player for training
        '''
        pass

    def set_learning(self, learning=False):
        '''
        updates self.learning to determine whether or not the player learns
        '''
        pass

    def learn(self, winner):
        '''
        informs the player about the outcome of a game
        '''

    def save(self):
        '''
        persists the player's internal state
        '''
        pass


class PlayerManual(Player):
    '''
    Player that asks a user playing token 'x' for input
    '''
    def get_next_move(self, board):
        '''
        returns position-tuple for token from user-input
        '''
        move = input(f'place a {self.token} at `i, j`: ')

        (i, j) = re.search('\s*(\d)\s*,\s*(\d)\s*', move).groups()

        return (int(i), int(j))

class PlayerRandom(Player):
    '''
    Player that makes random moves
    '''
    def get_next_move(self, board):
        '''
        returns position-tuple, a random free position
        '''
        return tuple(np.random.choice(board.get_free_positions()))


class PlayerEpsilonGreedy(Player):
    '''
    Player that uses RL with epsilon greedy to learn about moves
    '''
    def __init__(self, player_id, epsilon):
        super().__init__(player_id)

        self.epsilon = epsilon
        self.learning = False
        self.buffer = []

        self.knowledge_base = {}

    def set_learning(self, learning):
        '''
        updates the self.learning state
        '''
        assert type(learning) == bool, f'learning must be a bool, not {type(learning)}'

        self.learning = learning

    def reset(self):
        self.buffer = []

    def get_reward(self, winner):
        '''
        returns a reward (-1, 0, or 1) depending on the game outcome
        '''
        if winner == self.id: return 1
        if winner == 0: return 0

        return -1

    def get_next_move(self, board):
        board_hash = board.hash()

        # if this board state has never been seen, add it
        if board_hash not in self.knowledge_base:
            available_actions = board.get_free_positions().copy()

            self.knowledge_base[board_hash] = {
                'actions': available_actions,

                # use optimistic initialization for means
                'times': np.ones(len(available_actions)).astype('int'),
                'means': np.ones(len(available_actions))
            }

        # play epsilon greedy but update only at the end of the game
        state_data = self.knowledge_base[board_hash]

        if self.learning and np.random.random() < self.epsilon:
            # choose a random action...
            action_id = np.random.randint(len(state_data['actions']))
        else:
            # ...or the current best estimate
            action_id = np.argmax(state_data['means'])

        action = state_data['actions'][action_id]

        # keep track of which actions were taken in this game
        self.buffer.append({
            'state': board_hash,
            'action_id': action_id
        })

        return tuple(action)

    def learn(self, winner):
        '''
        updates the player's internal state after learning a new game outcome
        '''
        if not self.learning:
            print('not learning')
            return

        # evaluate the collected actions in light of the game outcome
        reward = self.get_reward(winner)

        for move_data in self.buffer:
            state_data = self.knowledge_base[move_data['state']]
            action_id = move_data['action_id']

            new_mean = self.compute_new_mean(state_data, action_id, reward)

            # update knowledgebase
            state_data['times'][action_id] += 1
            state_data['means'][action_id] = new_mean

    def compute_new_mean(self, state_data, action_id, reward):
        '''
        returns the new mean after observing the reward
        '''
        n = state_data['times'][action_id]
        old_mean = state_data['means'][action_id]

        return (n - 1) / n * old_mean + reward / n

    def save(self):
        print('save command received')
