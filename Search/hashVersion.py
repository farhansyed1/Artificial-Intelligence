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
        score_p0, score_p1 = node.state.get_player_scores()
        utility = 10 * (score_p0 - score_p1)

        p0_position = node.state.get_hook_positions()[0]
        p1_position = node.state.get_hook_positions()[1]
        fish_positions = node.state.get_fish_positions()
        fish_scores = node.state.get_fish_scores()

        # Closeness to high value fish
        for fish_id, fish_position in fish_positions.items():
            fish_value = fish_scores[fish_id]
            p0_dist, p1_dist = self.manhattdist(p0_position, p1_position, fish_position)

            if p0_dist < p1_dist:
                utility += fish_value / (1 + p0_dist)
            elif p1_dist < p0_dist:
                utility -= fish_value / (1 + p1_dist)  * 0.4

            if node.state.get_caught()[0] == fish_id:
                utility += fish_value * 2
        
        utility += 7 * node.depth
        return utility

    def manhattdist(self, p0, p1, fish):
        world_width = 20

        # P0
        p0_x, p0_y = p0
        fish_x, fish_y = fish
        p0_dist_x = min(abs(p0_x - fish_x), world_width - abs(p0_x - fish_x))
        p0_dist_y = abs(p0_y - fish_y)
        p0_dist = p0_dist_x + p0_dist_y

        # P1
        p1_x, p1_y = p1
        p1_dist_x = min(abs(p1_x - fish_x), world_width - abs(p1_x - fish_x))
        p1_dist_y = abs(p1_y - fish_y)
        p1_dist = p1_dist_x + p1_dist_y

        return p0_dist, p1_dist
 
    
    def alphabeta(self, node, depth, alpha, beta, start_time, maximizing_player):
        
        state_key = self.hashFunction(node.state, maximizing_player)

        if state_key in self.visited_states:
            if self.visited_states[state_key][0] >= depth:
                return self.visited_states[state_key][1]
        
        if time.time() - start_time > 0.058:
            raise TimeoutError
        
        children = node.compute_and_get_children()
        #children.sort(reverse=maximizing_player, key=self.evaluate_state)

        scores = []
        for i in range(len(children)):
            scores.append(self.evaluate_state(children[i]))

        move_order = sorted(range(len(scores)),
                            key=scores.__getitem__,
                            reverse=maximizing_player)

        sorted_children = [children[i] for i in move_order]

        if depth == 0 or not children:
            value = self.evaluate_state(node)
            self.visited_states[state_key] = (depth, value)
            return value
        
        if maximizing_player:
            value = -math.inf
            for child in sorted_children:
                value = max(value, self.alphabeta(child, depth - 1, alpha, beta, start_time, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  
        else:
            value = math.inf
            for child in sorted_children:
                value = min(value, self.alphabeta(child, depth - 1, alpha, beta, start_time, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break  
        
        self.visited_states[state_key] = (depth, value)
        return value
    
    def iterative_deepening_search(self, initial_tree_node, start_time, depth):
        best_move = 0  
        best_value = -math.inf
        
        alpha = -math.inf
        beta = math.inf

        children = initial_tree_node.compute_and_get_children()
        children.sort(reverse=True, key=self.evaluate_state)

        for child in children:
            move_value = self.alphabeta(child, depth, alpha, beta, start_time, True)
            if move_value > best_value:
                best_value = move_value
                best_move = child.move

        return best_move

    def hashFunction(self, state, player):
        score = state.player_scores[0] - state.player_scores[1]
        hello = 0
        if player:
            hello = 1
        
        return str(state.get_fish_positions()) + str(state.get_hook_positions()) + str(score) + str(hello)

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for p0 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        self.visited_states = dict()
        start_time = time.time()
        depth = 0
        best_move = 0

        while depth < 100:
            try:
                best_move = self.iterative_deepening_search(initial_tree_node, start_time, depth)
                depth += 1
            except TimeoutError:
                break

        return ACTION_TO_STR[best_move]