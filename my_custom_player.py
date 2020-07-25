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
       
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
            
        else:
            depth_limit = 3
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
                return self.base_score(state,play_id)
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
                return self.base_score(state,play_id)
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

    def base_score(self, state, player_id):
            own_loc = state.locs[player_id]
            opp_loc = state.locs[1 - player_id]
            own_liberties = state.liberties(own_loc)
            opp_liberties = state.liberties(opp_loc)
            return len(own_liberties) - len(opp_liberties)
        
    def intersect_score(self, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - 2*len(opp_liberties) + self.intersection(state, own_liberties, opp_liberties)
    
    def avoid_wall_score(self, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        distance = self.distance(state)
        if distance >= 2:
            return 2*len(own_liberties) - len(opp_liberties) + self.intersection(state, own_liberties, opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties) + self.intersection(state, own_liberties, opp_liberties)
        
    def combined_score(self, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        distance = self.distance(state)
        if distance >= 2:
            return 2*len(own_liberties) - 2*len(opp_liberties) + self.intersection(state, own_liberties, opp_liberties)
        else:
            return len(own_liberties) - 2*len(opp_liberties) + self.intersection(state, own_liberties, opp_liberties)
    
    def distance(self, state):
        """ minimum distance to the walls """
        own_loc = state.locs[state.player()]
        x_player, y_player = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)

        return min(x_player, _WIDTH + 1 - x_player, y_player, _HEIGHT - 1 - y_player)
   
    def intersection(self, state, own_liberties, opp_liberties):
        intersection = [x for x in own_liberties if x in opp_liberties]
        return len(intersection)