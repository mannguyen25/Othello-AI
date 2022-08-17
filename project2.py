"""
Group Name: Man Nguyen, Nick Adair, Lucy Zhang
Description: Implementations of several Othello game player classes. These include
             Minimax, Alpha-beta pruning, Quiescent Search, and Monte-Carlo
             Tree Search. The MiniMaxPlayer class also includes an unoptimized,
             partially broken version of Iterative Deepening Search.
             The Tournament Class is included, using alpha-beta pruning with our
             final heuistic for our round-robin touunament entry.
             A testing function is included to compare the Tournament class against
             various Homework 2 OthelloPlayers.
"""


import math
from re import I
from unittest import skip
from othello import *
from hw2 import ShortTermMaximizer, ShortTermMinimizer, MaximizeOpponentMoves,\
                MinimizeOpponentMoves, MaximizeDifference, MinimizeDifference
from collections import defaultdict
import random, sys
import _pickle as pickle
import csv

class MoveNotAvailableError(Exception):
    """Raised when a move isn't available."""
    pass

class OthelloTimeOut(Exception):
    """Raised when a player times out."""
    pass

class MCTState(OthelloState):
    """Expands OthelloState by including a hash function."""
    def __hash__(self):
        return hash(str(self.board))
    def __eq__(self, other):
        return self.board == other.board

def othello_to_mctstate(state):
    """Makes any OthelloState hashable and with an eq."""
    mcts = MCTState()
    mcts.board = state.board
    mcts.current = state.current
    mcts.move_number = state.move_number
    return mcts

class OthelloPlayer():
    """Parent class for Othello players."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass

class RandomPlayer(OthelloPlayer):
    """Plays a random move."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return random.choice(state.available_moves())

class HumanPlayer(OthelloPlayer):
    """Allows a human to play the game."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        available = state.available_moves()
        print("----- {}'s turn -----".format(state.current))
        print("Remaining time: {:0.2f}".format(remaining_time))
        print("Available moves are: ", available)
        move_string = input("Enter your move as 'r c': ")

        # Takes care of errant inputs and bad moves
        try:
            moveR, moveC = move_string.split(" ")
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_string, state.current))
            return self.make_move(state, remaining_time)

class MiniMaxPlayer(OthelloPlayer):
    """Player that runs the Minimax algorithm."""
    def __init__(self, color):
        OthelloPlayer.__init__(self, color)
        self.seen = set()

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        depth = state.move_number + 6
        isMAX = (state.move_number + 1) % 2
        # return self.mini_max(state, isMAX, depth)[0]
        # return self.iterative_deepening(state, isMAX)[0]
        return self.alpha_beta(state, -float('inf'), float('inf'), isMAX, depth)[0]
        # return self.mini_max_quiescent(othello_to_mctstate(state), isMAX, depth)[0]

    def mini_max(self, state, isMAX, end_depth, move=None):
        """The minimax algorithm."""
        if state.move_number == end_depth or state.game_over():
            return (None, heuristic1(state))
        if isMAX:
            value = -float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.mini_max(child, not isMAX, end_depth, move)[1]
                if child_val > value:
                    value = child_val
                    best_move = move
            return (best_move, value)
        else:
            value = float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.mini_max(child, not isMAX, end_depth, move)[1]
                if child_val < value:
                    value = child_val
                    best_move = move
            return (best_move, value)

    def alpha_beta(self, state, alpha, beta, isMAX, end_depth, move=None):
        """An implementation of alpha-beta pruning on the minimax algorithm."""
        if state.move_number == end_depth or state.game_over():
            return (None, heuristic1(state))
        if isMAX:
            value = -float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.alpha_beta(child, alpha, beta, not isMAX, end_depth, move)[1]
                if child_val > value:
                    value = child_val
                    best_move = move
                alpha = max(value, alpha)
                if alpha >= beta:
                    return (best_move, value)
            return (best_move, value)
        else:
            value = float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.alpha_beta(child, alpha, beta, not isMAX, end_depth, move)[1]
                if child_val < value:
                    value = child_val
                    best_move = move
                beta = min(value, beta)
                if beta < alpha:
                    return (best_move, value)
            return (best_move, value)

    def iterative_deepening(self, state, isMAX):
        """Iterative Deepening Search algorithm."""
        depth = 1
        self.seen = defaultdict(list)
        while depth < 6:
            # print("IDS depth", depth)
            new_state = copy.deepcopy(state)
            self.seen[depth] = sorted(self.seen[depth], key=lambda x:x[1])
            best_move, value = self.iterative_deepening_helper(new_state, -float('inf'), float('inf'), isMAX, state.move_number + depth)
            depth += 1
        # print("!!!",best_move, type(best_move))
        return (best_move, value)

    def iterative_deepening_helper(self, state, alpha, beta, isMAX, end_depth, move=None):
        """Recursive portion of IDS, which arranges pre-processed nodes."""
        if state.move_number == end_depth or state.game_over():
            value = heuristic3(state,isMAX)
            self.seen[state.move_number].append((move, value))
            return (None, heuristic3(state,isMAX))
        if isMAX:
            value = -float('inf')
            best_move = None
            available_moves = state.available_moves()
            if state.move_number in self.seen and self.seen[state.move_number] != []:
                available_moves = [pair[0] for pair in self.seen[state.move_number]]
            # print(self.seen,available_moves,state.available_moves())
            for move in available_moves :
                if move not in state.available_moves():
                    # print(move, state.available_moves(),state)
                    continue
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.iterative_deepening_helper(child, alpha, beta, not isMAX, end_depth, move)[1]
                if child_val > value:
                    value = child_val
                    best_move = move
                alpha = max(value, alpha)
                if alpha >= beta and not state.move_number == end_depth:
                    return (best_move, value)
            return (best_move, value)
        else:
            value = float('inf')
            best_move = None
            available_moves = state.available_moves()
            if state.move_number in self.seen and self.seen[state.move_number] != []:
                available_moves = reversed([pair[0] for pair in self.seen[state.move_number]])

            for move in available_moves:
                if move not in state.available_moves():
                    # print(move, state.available_moves(), state)
                    continue
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.iterative_deepening_helper(child, alpha, beta, not isMAX, end_depth, move)[1]
                if child_val < value:
                    value = child_val
                    best_move = move
                beta = min(value, beta)
                if beta < alpha:
                    return (best_move, value)
            return (best_move, value)

    def mini_max_quiescent(self, state, isMAX, end_depth, move=None):
        """The minimax algorithm with quiescent search."""
        if state.move_number == end_depth or state.game_over():
            return (None, heuristic5(state))
        if isMAX:
            value = -float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                if new_state in self.seen:
                    continue
                self.seen.add(new_state)
                board = new_state.board
                for _ in range(3):
                    rotated = copy.deepcopy(new_state)
                    board = rotate(rotated.board)
                    rotated.board = board
                    self.seen.add(rotated)
                child_val = self.mini_max(child, not isMAX, end_depth, move)[1]
                if child_val > value:
                    value = child_val
                    best_move = move
            return (best_move, value)
        else:
            value = float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                if new_state in self.seen:
                    continue
                self.seen.add(new_state)
                board = new_state.board
                for _ in range(3):
                    rotated = copy.deepcopy(new_state)
                    board = rotate(rotated.board)
                    rotated.board = board
                    self.seen.add(rotated)
                child_val = self.mini_max(child, not isMAX, end_depth, move)[1]
                if child_val < value:
                    value = child_val
                    best_move = move
            return (best_move, value)

def rotate(board):
    """Rotates the board by 90 degrees."""
    rotatedGrid = copy.deepcopy(board)
    reverse = board[::-1]
    for i in range (0, len(board)):
        for j in range (0, len(board)):
            rotatedGrid[i][j] = reverse[j][i]
    return rotatedGrid

class MCTSPlayer(OthelloPlayer):
    """Player that runs the MCTS algorithm."""

    def __init__(self, color):
        """Expands OthelloPlayer to include a tree."""
        assert color in ["black", "white"]
        self.color = color
        self.tree = self.make_tree()
        self.parent = None

    def make_tree(self):
        """Initalizes 'self.tree'."""
        with (open("tree.pickle", "rb")) as tree_file:
            return pickle.load(tree_file)

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        isMAX = (state.move_number + 1) % 2
        return self.MCTS(othello_to_mctstate(state),isMAX,remaining_time)

    def MCTS(self, state, isMAX, remaining_time):
        """Monte-Carlo Tree Search algorithm."""
        start_time = time.time()
        depth = state.move_number
        if depth < 4:
            search_time = 0
        elif depth < 14:
            search_time = 8
        elif depth < 30:
            search_time = 6
        else:
            search_time = 4
        while time.time() - start_time < search_time:
            states_involved, node, parent_state = self.descend(state)
            if node is None:
                break
            self.add_node(parent_state, node)
            winner = self.simulate(node)
            self.update(states_involved, winner)
        best_move, self.parent = self.find_best_move(state, isMAX)
        return best_move

    def descend(self, state, states_involved=[]):
        """Simluates a new game based on the UCB of nodes in the tree."""
        selection = None
        value = -float('inf')
        for move in state.available_moves():
            new_state = copy.deepcopy(state).apply_move(move)
            ucb = self.UCB(new_state)
            if ucb > value:
                selection = new_state
                value = ucb
        if selection is None:
            return states_involved, None, state
        states_involved.append(selection)
        if selection not in self.tree:
            return states_involved, selection, state
        return self.descend(selection, states_involved)

    def add_node(self, parent_state, node):
        """Adds a new node to the tree with 0 wins and 0 games."""
        self.tree[node] = [0, 0, parent_state]

    def simulate(self, state):
        """Randomly simulates a game from a specificed state."""
        new_state = copy.deepcopy(state)
        while not new_state.game_over():
            new_state = new_state.apply_move(random.choice(new_state.available_moves()))
        return new_state.winner()

    def update(self, states_involved, winner):
        """Updates the wins and games played values for nodes in the tree."""
        for state in states_involved:
            if winner == "black":
                self.tree[state][0] += 1
            elif winner == "draw":
                self.tree[state][0] += 0.5
            self.tree[state][1] += 1

    def UCB(self, state):
        """Calculates the UCB of a node in the tree."""
        if state not in self.tree:
            return 100
        wk, nk, parent_state = self.tree[state]
        if parent_state not in self.tree:
            parent_state = self.parent
        if nk == 0:
            return 100
        if state.current == 'black':
            return wk/nk + 1.1 * math.sqrt(math.log(self.tree[parent_state][1])/nk)
        return (nk-wk)/nk + 1.1 * math.sqrt(math.log(self.tree[parent_state][1])/nk)

    def move_value(self, state, isMAX):
        """Returns the value (games played) of a move."""
        if state not in self.tree or self.tree[state][1] < 10:
            return 0
        elif isMAX:
            return self.tree[state][0] / self.tree[state][1]
        return (self.tree[state][1] - self.tree[state][0]) / self.tree[state][1]

    def find_best_move(self, state, isMAX):
        """Finds the best move from the specified state based on the tree."""
        best_move = None
        value = -float('inf')
        for move in state.available_moves():
            new_state = copy.deepcopy(state).apply_move(move)
            new_value = self.move_value(new_state, isMAX)
            if new_value > value:
                best_move = move
                value = new_value
        return best_move, new_state

def build_tree():
    """Constructs a pre-built tree 'tree.pickle' focused on inital states."""
    """The current 'tree.pickle' contains approxiamtely 130,000 nodes, each
       with at least 50 games played on them."""
    tree = defaultdict(list)
    initial_state = MCTState()
    start_time = time.time()
    count = 0
    while time.time() - start_time < 14400:
        count += 1
        depth = 0
        state = copy.deepcopy(initial_state)
        states_involved = [(state, None)]
        while not state.game_over():
            depth += 1
            parent = copy.deepcopy(state)
            state = state.apply_move(random.choice(state.available_moves()))
            if depth < 30:
                states_involved.append((state, parent))
        final_state = state
        for state, parent in states_involved:
            if state not in tree:
                tree[state] = [0, 0, parent]
            if final_state.winner() == "black":
                tree[state][0] += 1
            elif final_state.winner() == "draw":
                tree[state][0] += 0.5
            tree[state][1] += 1
    print(f"Number of games: {count}")
    delete = []
    for key, values in tree.items():
        if values[1] < 50:
            delete.append(key)
    count = len(delete)
    for key in delete:
        del tree[key]
    print(f"Number states pruned: {count}")
    with open("tree.pickle", 'wb') as file_object:
        pickle.dump(tree, file_object, protocol=5)

def heuristic1(state):
    """Basic heuristic evaluation for game state utility."""
    return state.evaluation()

def heuristic2(state, isMAX, move):
    """Corner correcting heuristic evaluation for game state utility."""
    utility = state.evaluation()
    (r, c) = move.pair
    corner = [0, 7]
    corner_surroundings = [[0,1], [1,0], [1,1],
                           [0,6], [1,6], [1,7],
                           [6,0], [6,1], [7,1],
                           [6,6], [6,7], [7,6]]
    if r in corner and c in corner:
        if isMAX:
            utility += 10
        else:
            utility -= 10
    elif [r, c] in corner_surroundings:
        if isMAX:
            utility -= 10
        else:
            utility += 10
    return utility

def heuristic3(state, isMax, parent):
    """Mobility based heuristic evaluation for game state utility."""
    return len(state.available_moves()) if isMax else len(parent.available_moves())

def heuristic4(state, isMax, move):
    """Position-based heurisitc on static point assignments.
    Corners heavily prioritized, edges + corners are good, anything one position
    away from the edges is bad, and cells on the digonal in adjacent to c."""
    weights = [[10, -8, 5, 5, 5, 5, -8, 10],
               [-8, -10, -5, -5, -5, -5, -10, -8],
               [5, -5, 2, 0, 0, 2, -5, 5],
               [5, -5, 0, 2, 2, 0, -5, 5],
               [5, -5, 0, 2, 2, 0, -5, 5],
               [5, -5, 2, 0, 0, 2, -5, 5],
               [-8, -10, -5, -5, -5, -5, -10, -8],
               [10, -8, 5, 5, 5, 5, -8, 10]]
    (r, c) = move.pair
    return weights[r][c] if isMax else -weights[r][c]

def heuristic5(state, isMax, move, parent):
    """Heuristic for corner weights and mobility."""
    return heuristic3(state, isMax, parent) +  2*heuristic2(state, isMax, move)

def heuristic6(state, isMax, move):
    """Depth based combination heuristic."""
    # at depth five current state is white if playing black
    # opening
    if state.move_number < 15:
        return state.evaluation()
        # return int((1 - abs(state.evaluation()/(state.count('black') + state.count('white'))))*100)
    # mid-game
    elif state.move_number < 40:
        return heuristic5(state, isMax, move)
    # end game
    else:
        return state.evaluation()

def heuristic7(state, isMax, parent):
    """Heuristic based on minimizing change."""
    # in the state of white
    if isMax:
        current = state.count(opposite_color(state.current))
        return current - parent.count(opposite_color(state.current))
    current = state.count(state.current)
    return current - parent.count(opposite_color(state.current))

def heuristic8(state, isMax, move, parent):
    """Minimize flanking positions."""
    r,c = move.pair
    flank_count = 0
    if isMax:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr==0 and dc==0:
                    continue
                if state.flanking(r + dr, c + dc,
                                    dr, dc, opposite_color(state.current),
                                    state.current):
                    flank_count += 1
        return -flank_count
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr==0 and dc==0:
                continue
            if parent.flanking(r + dr, c + dc,
                                dr, dc, state.current,
                                opposite_color(state.current)):
                flank_count += 1
    return -flank_count

def mega_heuristic(state, isMax, move, parent, color):
    """The best heuristic."""
    if color == 'black':
        return (heuristic2(state, isMax, move) + heuristic3(state, isMax, parent))
    else:
        return (2 * heuristic2(state, isMax, move) + heuristic3(state, isMax, parent) + 0.5 * heuristic7(state, isMax, parent))

class TournamentPlayer(OthelloPlayer):
    """Player for the round-robin tounrment."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        depth = state.move_number + 5
        if state.move_number >= 36:
            depth = state.move_number + 7
        elif state.move_number >= 48:
            depth = state.move_number + 11
        isMAX = (state.move_number + 1) % 2
        return self.alpha_beta(state, -float('inf'), float('inf'), isMAX, depth)[0]

    def alpha_beta(self, state, alpha, beta, isMAX, end_depth, move=None, parent=None):
        """An implementation of alpha-beta pruning on the minimax algorithm."""
        if state.move_number == end_depth or state.game_over():
            return (None, mega_heuristic(state, isMAX, move, parent, self.color))
        if isMAX:
            value = -float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.alpha_beta(child, alpha, beta, not isMAX, end_depth, move, parent=new_state)[1]
                if child_val > value:
                    value = child_val
                    best_move = move
                alpha = max(value, alpha)
                if alpha >= beta:
                    return (best_move, value)
            return (best_move, value)
        else:
            value = float('inf')
            best_move = None
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                child = new_state.apply_move(move)
                child_val = self.alpha_beta(child, alpha, beta, not isMAX, end_depth, move, parent=new_state)[1]
                if child_val < value:
                    value = child_val
                    best_move = move
                beta = min(value, beta)
                if beta < alpha:
                    return (best_move, value)
            return (best_move, value)

################################################################################

def main():
    """Plays the game."""
    black_player = ShortTermMaximizer("black")
    white_player = TournamentPlayer("white")

    game = OthelloGame(black_player, white_player, verbose=True)

    winner = game.play_game_timed()

    # ###### Use this method if you want to use a HumanPlayer
    # winner = game.play_game()

    # if not game.verbose:
    #     print("Winner is", winner)

if __name__ == "__main__":
    main()
