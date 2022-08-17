"""
Group Name: Man Nguyen, Nick Adair, Lucy Zhang
Description: Implementations of several Othello game player classes, including a random Player that makes a random move given their available moves),
             players that minimize and maximize the number of its color pieces on the board after its turn,
             players that minimize and maximize the number of moves its opponnent has after its move,
             and players that minimize and maximize the difference between black and white pieces.
             Implementation of a testing/data collection function that matches each player with every other player and
             collects data about who wins, how many pieces of each color for wins and losses.
"""

from othello import *
from itertools import permutations
import random


class MoveNotAvailableError(Exception):
    """Raised when a move isn't available. Do not change."""
    pass


class OthelloPlayer():
    """Parent class for Othello players. Do not change."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass


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
                raise MoveNotAvailableError  # Indicates move isn't available
        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_string, state.current))
            return self.make_move(state, remaining_time)


class RandomPlayer(OthelloPlayer):
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""

        return random.choice(state.available_moves())


class ShortTermMaximizer(OthelloPlayer):
    """ Player that maximizes the number of its color pieces on
        the board after their move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. remaining_time is how much
        time this player has to finish the game."""

        moves = state.available_moves()
        # return only move if there are no other options
        if len(moves) < 2:
            return moves[0]
        # set the maximized move as our initial comparison
        max_move = moves[0]
        max_state = copy.deepcopy(state).apply_move(max_move)
        # count the number of pieces and choose the one that maximizes the count
        # for the respective player
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            if new_state.count(move.player) > max_state.count(move.player):
                max_move = move
                max_state = new_state
        return max_move


class ShortTermMinimizer(OthelloPlayer):
    """ Player that minimizes the number of its color pieces on
        the board after their move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. remaining_time is how much
        time this player has to finish the game."""

        moves = state.available_moves()
        if len(moves) < 2:
            return moves[0]
        min_move = moves[0]
        min_state = copy.deepcopy(state).apply_move(min_move)
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            if new_state.count(move.player) < min_state.count(move.player):
                min_move = move
                min_state = new_state
        return min_move


class MaximizeOpponentMoves(OthelloPlayer):
    """ Player that maximizes the number of the opponents moves after its move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. remaining_time is how much
        time this player has to finish the game."""

        moves = state.available_moves()
        if len(moves) < 2:
            return moves[0]
        max_move = moves[0]
        max_state = copy.deepcopy(state).apply_move(max_move)
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            if len(new_state.available_moves()) > len(max_state.available_moves()):
                max_move = move
                max_state = new_state
        return max_move


class MinimizeOpponentMoves(OthelloPlayer):
    """ Player that minimizes the number of the opponents moves after its move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. remaining_time is how much
        time this player has to finish the game."""

        moves = state.available_moves()
        if len(moves) < 2:
            return moves[0]
        min_move = moves[0]
        min_state = copy.deepcopy(state).apply_move(min_move)
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            if len(new_state.available_moves()) < len(min_state.available_moves()):
                min_move = move
                min_state = new_state
        return min_move


class MaximizeDifference(OthelloPlayer):
    """ Player that minimizes the number of the opponents moves after its move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. remaining_time is how much
        time this player has to finish the game."""

        moves = state.available_moves()
        if len(moves) < 2:
            return moves[0]
        max_move = moves[0]
        max_state = copy.deepcopy(state).apply_move(max_move)
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            if new_state.evaluation() > max_state.evaluation():
                max_move = move
                max_state = new_state
        return max_move


class MinimizeDifference(OthelloPlayer):
    """ Player that minimizes the number of the opponents moves after its move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. remaining_time is how much
        time this player has to finish the game."""

        moves = state.available_moves()
        if len(moves) < 2:
            return moves[0]
        min_move = moves[0]
        min_state = copy.deepcopy(state).apply_move(min_move)
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            if new_state.evaluation() < min_state.evaluation():
                min_move = move
                min_state = new_state
        return min_move


def test_wrapper():
    """Wrapper to test multiple combinations on Players."""

    players = [(RandomPlayer, "RandomPlayer"),
               (ShortTermMaximizer, "ShortTermMaximizer"),
               (ShortTermMinimizer, "ShortTermMinimizer"),
               (MaximizeOpponentMoves, "MaximizeOpponentMoves"),
               (MinimizeOpponentMoves, "MinimizeOpponentMoves"),
               (MaximizeDifference, "MaximizeDifference"),
               (MinimizeDifference, "MinimizeDifference")]
    pairs = permutations(players, 2)
    with open("data.csv", 'w') as file_object:
        print("Black, White, Winner, Wins, Draws, White Win Pieces, White Loss Pieces", file=file_object)
        for pair in pairs:
            random = True
            black_player = pair[0][0]("black")
            black_name = pair[0][1]
            white_player = pair[1][0]("white")
            white_name = pair[1][1]
            if black_name != "RandomPlayer" and white_name != "RandomPlayer":
                random = False
            # print("------------------------------------", file=file_object)
            # print(f"Black: {black_name}", file=file_object)
            # print(f"White: {white_name}", file=file_object)
            print(f"{black_name}, {white_name}", end='', file=file_object)
            test(black_player, white_player, random, file_object)


def test(black_player, white_player, random, file_object=None):
    """Data collection function."""
    iterations = 1
    if random:
        iterations = 1000
    winner_list = []
    winner_pieces = []
    loser_pieces = []
    for i in range(iterations):
        game = OthelloGame(black_player, white_player, verbose=True)
        winner = game.play_game()
        winner_list.append(winner)
        if winner != "draw":
            winner_pieces.append(game.board.count(winner_list[-1]))
            loser_pieces.append(game.board.count(opposite_color(winner_list[-1])))
        else:
            winner_pieces.append(None)
            loser_pieces.append(None)

    black_win_pieces = []
    black_loss_pieces = []
    white_win_pieces = []
    white_loss_pieces = []
    num_draws = 0
    for i in range(len(winner_list)):
        if winner_list[i] == "white":
            white_win_pieces.append(winner_pieces[i])
            black_loss_pieces.append(loser_pieces[i])
        elif winner_list[i] == "black":
            black_win_pieces.append(winner_pieces[i])
            white_loss_pieces.append(loser_pieces[i])
        else:
            num_draws += 1

    if len(white_win_pieces) > len(black_win_pieces):
        winner = "white"
        num_wins = len(white_win_pieces)
    else:
        winner = "black"
        num_wins = len(black_win_pieces)

    # print(f"{winner} wins {num_wins} out of {iterations}", file=file_object)
    # print(f"The number of draws is {num_draws}", file=file_object)

    if len(black_win_pieces) != 0:
        avg_black_win_pieces = sum(black_win_pieces) / len(black_win_pieces)
        # print(f"Average Black Pieces For Wins: {avg_black_win_pieces}", file=file_object)
    if len(black_loss_pieces) != 0:
        avg_black_loss_pieces = sum(black_loss_pieces) / len(black_loss_pieces)
        # print(f"Average Black Pieces For Losses: {avg_black_loss_pieces}", file=file_object)
    if len(white_win_pieces) != 0:
        avg_white_win_pieces = sum(white_win_pieces) / len(white_win_pieces)
        # print(f"Average White Pieces For Wins: {avg_white_win_pieces}", file=file_object)
    if len(white_loss_pieces) != 0:
        avg_white_loss_pieces = sum(white_loss_pieces) / len(white_loss_pieces)
        # print(f"Average White Pieces For Losses: {avg_white_loss_pieces}", file=file_object)

    if random:
        print(f", {winner}, {num_wins}, {num_draws}, {avg_white_win_pieces}, {avg_white_loss_pieces}", file=file_object)
    else:
        print(f", {winner}, {num_wins}, {num_draws}, {None}, {None}", file=file_object)

################################################################################

def main():
    """Plays the game. You'll likely want to make a new main function for
    playing many games using your players to gather stats."""

    # black_player = RandomPlayer("black")
    # white_player = RandomPlayer("white")
    #
    # game = OthelloGame(black_player, white_player, verbose=True)
    #
    # ###### Use this method if you want to play a timed game. Doesn't work with HumanPlayer
    # # winner = game.play_game_timed()
    #
    # ###### Use this method if you want to use a HumanPlayer
    # winner = game.play_game()
    #
    #
    # if not game.verbose:
    #     print("Winner is", winner)

    game = OthelloGame(black_player, white_player, verbose=True)

    ###### Use this method if you want to play a timed game. Doesn't work with HumanPlayer
    # winner = game.play_game_timed()

    ###### Use this method if you want to use a HumanPlayer
    winner = game.play_game()
    if not game.verbose:
        print("Winner is", winner)

    # test_wrapper()


if __name__ == "__main__":
    main()
