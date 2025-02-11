

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random


torch.manual_seed(0)

class Connect4:
    def __init__(self):
        self.row_count = 6
        self.colum_count = 7
        self.action_size = self.colum_count
        self.in_a_row = 4

    def __repr__(self):
        return "Connect4"

    def get_init_state(self):
        return np.zeros((self.row_count, self.colum_count))

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= self.row_count
                        or c < 0
                        or c >= self.colum_count
                        or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
                count(1, 0) >= self.in_a_row - 1
                or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1
                or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1
                or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1
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

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def play(self, state, player, model):
        while True:
            print(state)
            if player == 1:
                valid_moves = self.get_valid_moves(state)
                print("Valid moves:", [i for i in range(self.action_size) if valid_moves[i] == 1])
                action = int(input(f"Player {player}, enter your move (0-8): "))

                if valid_moves[action] == 0:
                    print("Action not valid. Try again.")
                    continue
            else:
                neutral_state = self.change_perspective(state, player)
                mcts_probs = mcts.search(neutral_state, model)
                action = np.argmax(mcts_probs)

            state = self.get_next_state(state, action, player)
            value, is_terminal = self.get_value_and_terminate(state, action)

            if is_terminal:
                print(state)
                if value == 1:
                    print(f"Player {player} won!")
                else:
                    print("It's a draw!")
                break

            player = self.get_opponent(player)

class Engine_test:
    def __init__(self, game, state, args, player, model_1, model_2):
        self.game = game
        self.args = args
        self.player = player
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_win_counter = 0
        self.model_2_win_counter = 0
        self.draw_counter = 0
        self.state = state

    def count_win(self, player):
        if player == 1:
            self.model_1_win_counter += 1

        elif player == -1:
            self.model_2_win_counter += 1



    def show_results(self):
        print(f"engine 1 wins: {self.model_1_win_counter}")
        print(f"engine 2 wins: {self.model_2_win_counter}")
        print("draws:", 100-(self.model_2_win_counter + self.model_1_win_counter))

    def engine_play(self):
        start_player = self.player
        for game in range(self.args['num_engine_games']):
            state = self.game.get_init_state()
            start_player = self.game.get_opponent(start_player)
            player = start_player


            while True:
                if player == 1:
                    neutral_state = self.game.change_perspective(state, player)
                    mcts_probs = mcts.search(neutral_state, self.model_1)
                    action = np.argmax(mcts_probs)
                else:
                    neutral_state = self.game.change_perspective(state, player)
                    mcts_probs = mcts.search(neutral_state, self.model_2)
                    action = np.argmax(mcts_probs)

                state = self.game.get_next_state(state, action, player)
                value, is_terminal = self.game.get_value_and_terminate(state, action)

                if is_terminal:
                    if value == 1:
                        self.count_win(player)
                        print(f"Player {player} won")
                    else:
                        print("draw")
                    break

                player = self.game.get_opponent(player)










class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.colum_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.colum_count, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (1 + child.visit_count)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
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

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = ((1-self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']]* self.game.action_size, size=policy.shape[0]))

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(spg.state)
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, spg.state, visit_count=1)

            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

                expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

                if len(expandable_spGames) >0:
                    states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                    )
                    policy = torch.softmax(policy, axis=1).cpu().numpy()
                    value = value.cpu().numpy()

                for i, mappingIdx in enumerate(expandable_spGames):
                    spg_policy, spg_value = policy[i], value[i]
                    node = spGames[mappingIdx].node

                    valid_moves = self.game.get_valid_moves(node.state)
                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)

                    node.expand(spg_policy)
                    node.backpropagate(spg_value)

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args


    @torch.no_grad()
    def search(self, state, model):
        root = Node(self.game, self.args, state, visit_count=1)
        policy, _ = model(
            torch.tensor(self.game.get_encoded_state(state), device=model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = ((1-self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']]* self.game.action_size))

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves

        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)  # Ensure it sums to 1
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminate(spg.state, action)
                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory



    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) -1, batchIdx+self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            state = torch.tensor(state,dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []


            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                print(f"iteration: {iteration}, game: {selfPlay_iteration}")


            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt" )
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

class SPG:
    def __init__(self, game):
        self.state = game.get_init_state()
        self.memory = []
        self.root = None
        self.node = None

game = Connect4()
player = 1
state = game.get_init_state()

args = {
    'C': 2,
    'num_searches': 1000,
    'num_iterations' : 8,
    'num_selfPlay_iterations' : 300,
    'num_parallel_games' : 15,
    'num_epochs' : 4,
    'batch_size' : 64,
    'temperature' : 1.25,
    'dirichlet_epsilon' : 0.25,
    'dirichlet_alpha' : 0.3,
    'num_engine_games' : 100
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = ResNet(game, 9, 128, device=torch.device("cpu"))
model_2 = ResNet(game, 9, 128, device=torch.device("cpu"))
state_dict_1 = torch.load("model_7_Connect4.pt", weights_only=True)
model_1.load_state_dict(state_dict_1)
model_1.eval()


state_dict_2 = torch.load("model_7_Connect4_q-test.pt", weights_only=True)
model_2.load_state_dict(state_dict_2)
model_2.eval()

mcts = MCTS(game, args)
mcts_train = MCTSParallel(game, args, model_1)

engine_test = Engine_test(game, state, args, player, model_1, model_2)
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001, weight_decay=0.0001)
alphazero = AlphaZeroParallel(model_1, optimizer, game, args)


game.play(state, player, model_2)
