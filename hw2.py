"""
Group Name: Man Nguyen, Nick Adair, Lucy Zhang
Description: 
Put a nice docstring here!
"""

from othello import *
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
                raise MoveNotAvailableError # Indicates move isn't available

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
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""

        return random.choice(state.available_moves())
    
class ShortTermMinimizer(OthelloPlayer):
    """ Player that minimizes the number of their color pieces on 
        the board after that move """
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        moves = state.available_moves()
        if len(moves) < 2:
            return moves[0]
        min_move = moves[0]
        min_state = copy.deepcopy(state).apply_move(min_move)
        for move in moves[1:]:
            new_state = copy.deepcopy(state)
            new_state = new_state.apply_move(move)
            print(min_state.count(move.player), new_state.count(move.player))
            if new_state.count(move.player) < min_state.count(move.player):
                min_move = move
                min_state = new_state
        return min_move

class MaximizeOpponentMoves(OthelloPlayer):
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""

        return random.choice(state.available_moves())

class MinimizeOpponentMoves(OthelloPlayer):
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""

        return random.choice(state.available_moves())


################################################################################

def main():
    """Plays the game. You'll likely want to make a new main function for
    playing many games using your players to gather stats."""

    black_player = ShortTermMinimizer("black")
    white_player = RandomPlayer("white")

    game = OthelloGame(black_player, white_player, verbose=True)

    ###### Use this method if you want to play a timed game. Doesn't work with HumanPlayer
    # winner = game.play_game_timed()

    ###### Use this method if you want to use a HumanPlayer
    winner = game.play_game()


    if not game.verbose:
        print("Winner is", winner)


if __name__ == "__main__":
    main()
