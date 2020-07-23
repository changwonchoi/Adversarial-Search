from isolation.isolation import _WIDTH, _HEIGHT

def alpha_beta_search(gameState, depth):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """

    def min_value(gameState, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(0)
        
        if depth <= 0 :
            player_loc = gameState.locs[gameState.player()]
            opponent_loc = gameState.locs[1-gameState.player()]
            # divide the players liberties by the result of wall, so if current location is near wall, it won't get as many points
            return len(gameState.liberties(player_loc)) / (wall(gameState) + 1) - len(gameState.liberties(opponent_loc))
        
        v = float("inf")
        for a in gameState.actions():
            v = min(v, max_value(gameState.result(a), alpha, beta, depth-1))
            if v <= alpha: return v
            beta = min(beta,v)
        return v

    def max_value(gameState, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(0)
        
        if depth <= 0 :
            player_loc = gameState.locs[gameState.player()]
            opponent_loc = gameState.locs[1-gameState.player()]
            # divide the players liberties by the result of wall, so if current location is near wall, it won't get as many points
            return len(gameState.liberties(player_loc)) / (wall(gameState) + 1) - len(gameState.liberties(opponent_loc))
        
        v = float("-inf")
        for a in gameState.actions():
            v = max(v, min_value(gameState.result(a), alpha, beta, depth-1))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v
    
    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    for a in gameState.actions():
        v = min_value(gameState.result(a), alpha, beta, depth)
        alpha = max(alpha, v)
        if v > best_score:
            best_score = v
            best_move = a
    return best_move

def wall(state):
    loc = state.locs[state.player()]
    # find out players position in terms of x,y coordinates
    x = loc % (_WIDTH + 2)
    y = loc // (_WIDTH + 2)
    
    # determine how close the player is to the walls
    proximity = 0
    if x <= 3 or x >= (_WIDTH-2):
        proximity += 1
    if y <= 2 or y >= (_HEIGHT-3):
        proximity += 1
    return proximity
