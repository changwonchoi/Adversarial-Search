import random
from isolation.isolation import _WIDTH, _HEIGHT
from sample_players import DataPlayer

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        
        # The player chooses random move at first.
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
            
        else:
            depth_limit = 4
            move = None
            for depth in range(1, depth_limit+1):
                move = self.alpha_beta_search(state, self.player_id, depth)
            self.queue.put(move)
        
    def alpha_beta_search(self, state,play_id,depth=3):
        """ Return the move along a branch of the game tree that
        has the best possible value.
        """
        def min_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(play_id)
            if depth <= 0:
                return self.combined_score(state,play_id)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth-1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(play_id)
            if depth <= 0:
                return self.combined_score(state,play_id)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth-1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            value = min_value(state.result(action), alpha, beta, depth-1)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move

    """
    Basic heuristic function implemented in the class.
    Compares the number of my liberties and the opponent's liberties
    """
    def base_score(self, state, player_id):
        my_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        my_liberties = state.liberties(my_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(my_liberties) - len(opp_liberties)
        
    """
    Heuristic function that aims to limit the opponent's move by blocking it.
    The function puts a heavier weight on the opponent's liberties,
    so that the custom player will try to choose a move where the the opponent has lesser moves.
    Also, my adding the intersection function, the custom player seeks to choose a move
    that could have been one of the opponent's available moves, thus the player can block the opponent's  next move.
    """
    def intersect_score(self, state, player_id):
        my_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        my_liberties = state.liberties(my_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(my_liberties) - 2*len(opp_liberties) + self.intersection(state, my_liberties, opp_liberties)
    
    
    """
    Heuristic to stay away from the wall.
    Through some research, I found out that going near the wall traps the player.
    For example, if a player is 1 unit away from a wall, the wall limits two moves of the player.
    If the player is right by the wall, the wall limits 4 moves of the player.
    If the player is 1 unit away from walls on both x and y coordinates, the walls limit four moves of the player.
    Finally if the player is at the corner of the board, the player only has maximum of three moves.
    Therefore this heuristic puts more weight on the case where the player is at least 2 units away from each wall,
    so that the player is more likely to stay away from the wall.
    """
    def avoid_wall_score(self, state, player_id):
        my_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        my_liberties = state.liberties(my_loc)
        opp_liberties = state.liberties(opp_loc)
        if self.distance(state):
            return 2*len(my_liberties) - len(opp_liberties)
        else:
            return len(my_liberties) - len(opp_liberties)
    
    """
    Heuristic that combines both intersection and avoid-wall heuristics.
    I thought of different ways to combine the two heuristics, but I think this one makes most sense.
    When the number of my_liberties is multiplied by two, coustom player is more likely to play to save its moves.
    In contrast, when the number of opp_liberties is multiplied by two, custom player is more likely to play to get rid of the opponent's moves.
    Intersection heuristic aims to limit opponent's move, and avoid-wall heuristic aims to avoid wall, so that the player will have more available moves.
    Therefore, I put more weight on the move that avoids walls, but at the same time I kept the weight on opponent's liberties,
    so that the player will still try to block the opponent's move while trying to avoid the wall as well.
    """
    def combined_score(self, state, player_id):
        my_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        my_liberties = state.liberties(my_loc)
        opp_liberties = state.liberties(opp_loc)
        if self.distance(state):
            return 2*len(my_liberties) - 2*len(opp_liberties) + self.intersection(state, my_liberties, opp_liberties)
        else:
            return len(my_liberties) - 2*len(opp_liberties) + self.intersection(state, my_liberties, opp_liberties)
    
    """
    Helper function to calculate the players distance to each wall, and determine if the player is close to the wall or not.
    """
    def distance(self, state):
        """ minimum distance to the walls """
        my_loc = state.locs[state.player()]
        x = my_loc // (_WIDTH + 2)
        y = my_loc % (_WIDTH + 2)
        if min(x, _WIDTH + 1 - x, y, _HEIGHT - 1 - y) >= 2:
            return True
        else:
            return False
   
    """
    Helper function to calculate how many of the player's moves interset with the opponent's moves.
    """
    def intersection(self, state, my_liberties, opp_liberties):
        intersection = [action for action in my_liberties if action in opp_liberties]
        return len(intersection)