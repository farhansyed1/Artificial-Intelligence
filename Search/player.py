#!/usr/bin/env python3
import random
import math
import time
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})
    
    def evaluate_state(self, node):
        score_p0, score_p1 = node.state.get_player_scores() #obtain the current scores of players
        p0_position = node.state.get_hook_positions()[0]   
        p1_position = node.state.get_hook_positions()[1] 

        fish_positions = node.state.get_fish_positions() 
        fish_scores = node.state.get_fish_scores()

        utility = score_p0 - score_p1 #initial utility calculation. A positive value means player 0 is leading.

        # Manhattan distances
        for fish_id, fish_position in fish_positions.items(): #Iterate over each fish on the board.
            fish_value = fish_scores[fish_id]

            #calculate the manhattan distance from each player to the current fish
            p0_dist, p1_dist = self.manhattdist(p0_position, p1_position, fish_position) 

            
            if p0_dist < p1_dist: #player 0 is closer to the fish than player 1
                utility += fish_value / (1 + p0_dist)  
            elif p1_dist < p0_dist: #player 1 is closer to the fish than player 0
                utility -= fish_value / (1 + p1_dist) 

        return utility
    
    def manhattdist(self, p0, p1, fish):
        world_width = 20

        # P0
        p0_x, p0_y = p0
        fish_x, fish_y = fish
        p0_dist_x = min(abs(p0_x - fish_x), world_width - abs(p0_x - fish_x)) #Chooses the shortest horisontal path, direct or wrap-around from p0 to fish
        p0_dist_y = abs(p0_y - fish_y) # vertical distance from p0 to fish
        p0_dist = p0_dist_x + p0_dist_y # total manhattan distance from p0 to fish

        # P1
        p1_x, p1_y = p1
        p1_dist_x = min(abs(p1_x - fish_x), world_width - abs(p1_x - fish_x))
        p1_dist_y = abs(p1_y - fish_y)
        p1_dist = p1_dist_x + p1_dist_y

        return p0_dist, p1_dist
 
    
    def alphabeta(self, node, depth, alpha, beta, start_time, maximizing_player):
            node.compute_and_get_children()
            #If the maximum search depth has been reached
            #If the node has no children (terminal node)
            if depth == 0 or len(node.children) == 0: 
                return self.evaluate_state(node) #evaluate utility value of the current node (game state)
            elif time.time() - start_time > 0.057:
                raise TimeoutError
            
            if maximizing_player:
                value = -math.inf #init value, worst possible score for the maximizing player
                for child in node.children:
                    value = max(value, self.alphabeta(child, depth - 1, alpha, beta, start_time, False)) #recursive step, find highest utility value found among the children 
                    alpha = max(alpha, value) #Represents the highest/best utility value that the maximizing player can guarantee at this point.
                    if beta <= alpha: #Stops evaluating remaining child nodes because the minimizing player will avoid this branch, as there's already a better option available
                        break  
                return value #Returns the best utility value found for the maximizing player at this game state
            else: #minimizing player's turn
                value = math.inf
                for child in node.children:
                    value = min(value, self.alphabeta(child, depth - 1, alpha, beta, start_time, True)) #recursive step, keep track of the lowest utility value found among the children
                    beta = min(beta, value) #Represents the best/lowest utility value that the minimizing player can guarantee at this point
                    if beta <= alpha: #If true, prune
                        break  
                return value
    
    def iterative_deepening_search(self, initial_tree_node, max_time):
        start_time = time.time() #current time when the search begins
        best_move = 0  
        depth = 0
        best_value = -math.inf #worst possible score for the maximizing player

        while True:
            try:
                alpha = -math.inf #best score that the maximizing player can guarantee
                beta = math.inf #best score that the minimizing player can guarantee

                initial_tree_node.compute_and_get_children() #all possible moves (child nodes) from the current game state
                for child in initial_tree_node.children:
                    #recursively evaluate the child nodes up to the specified depth and get utility for that game state
                    move_value = self.alphabeta(child, depth, alpha, beta, start_time, True)
                    if move_value > best_value: #if a better value is found, update it 
                        best_value = move_value
                        best_move = child.move #store the move of the specific child node as the best move
                    alpha = max(alpha, best_value) #best/highest utility value found so far for the maximizing player
                depth += 1

                if time.time() - start_time > max_time:
                    break
            except TimeoutError:
                break

        return best_move


    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        max_time = 0.057
        best_move = self.iterative_deepening_search(initial_tree_node, max_time)
        return ACTION_TO_STR[best_move]

