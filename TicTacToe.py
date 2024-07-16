import numpy as np
import math

class tictactoe:
    def __init__(self):
        self.row_count = 3
        self.colum_count = 3
        self.action_size = self.row_count * self.colum_count

    def get_init_state(self):
        return np.zeros((self.row_count, self.colum_count))

    def get_next_state(self, state, action, player):
        row = action // self.row_count
        colum = action % self.colum_count
        new_state = state.copy()  # Make a copy to avoid modifying the original state
        new_state[row, colum] = player
        return new_state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False
        row = action // self.row_count
        colum = action % self.colum_count
        player = state[row, colum]

        return (
            np.sum(state[row, :]) == player * self.colum_count
            or np.sum(state[:, colum]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.colum_count
        )

    def get_value_and_terminate(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminate(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminate(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            rollout_player = self.game.get_opponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


game = tictactoe()
player = 1
state = game.get_init_state()

args = {
    'C': 1.41,
    'num_searches': 1000
}
mcts = MCTS(game, args)

while True:
    print(state)
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("Valid moves:", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"Player {player}, enter your move (0-8): "))

        if valid_moves[action] == 0:
            print("Action not valid. Try again.")
            continue
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = game.get_next_state(state, action, player)
    value, is_terminal = game.get_value_and_terminate(state, action)

    if is_terminal:
        print(state)
        if value == 1:
            print(f"Player {player} won!")
        else:
            print("It's a draw!")
        break

    player = game.get_opponent(player)
