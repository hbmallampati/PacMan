# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """

    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"

    #from game import Directions

    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    #print("Actions from start state:",problem.getActions(problem.getStartState()))
    #print("Getting Result", problem.getResult(problem.getStartState(), problem.getActions(problem.getStartState()[2])))
    
    #get the coordinates of the start state
    start = problem.getStartState()
    #print(start)
    #visited contains nodes that have been removed from the tree and the direction from which they have been obtained
    visited = {}

    #solution contains the sequence of directions for Pacman to get to the goal state
    solution = []

    #node will contain of the co-ordinate values, it's direction, and the cost. start is the root node.
    node = (start, 'Start Node', 0)
    queue = []
    queue.append(node)
    #print(queue)

    #the start state doesn't have a direction from where it's arrived
    visited[start] = 'Start Node'

    #parents will consist of the node and their parents
    parents = {}

    #if start state is goal state
    if problem.goalTest(start):
        return solution

    #stores the cost
    cost = 0

    #loop while the goal state has not been reached and the queue is not empty
    goal = False
    while queue != [] and goal != True:

        node = queue.pop()
        #store the element and it's direction
        visited[node[0]] = node[1]
        #if the element is the goal state then exit the loop
        if problem.goalTest(node[0]):
            #print('Worked')
            leaf_node = node[0]
            goal = True
            break
        #else expand the node
        for child in problem.getActions(node[0]):
            #if the child node has not been visited yet
            if problem.getResult(node[0], child) not in visited.keys():
                #store the child node and it's parent
                parents[problem.getResult(node[0], child)] = node[0]
                cost += 1
                queue.append((problem.getResult(node[0], child), child, cost))
                #print(queue)
    print(parents)
    #finding and storing the path
    while(leaf_node in parents.keys()):
        #find the parent
        parent_node = parents[leaf_node]
        #add the direction to the solution
        solution.insert(0, visited[leaf_node])
        #go to the previous node
        leaf_node = parent_node

    #successors = []

    #for i in range(len(problem.getActions(start))):
    #    successors.append((problem.getResult(start, problem.getActions(start)[i]), problem.getActions(start)[i]))
    
    #print(successors)
    print(solution)
    return solution

    '''startState = problem.getStartState()
    #print(startState)
    nodesExplored = []
    frontierNodes = util.Queue()
    node = Node(startState, None, None, 0)
    frontierNodes.push(node)
    actionRet = {}
    solution = []

    while not frontierNodes.isEmpty():
        node = frontierNodes.pop()
        currNode = node.state
        action = problem.getActions(currNode)

        if not currNode in nodesExplored:
            nodesExplored.append(currNode)
            #print('Frontier Nodes travelled: ', nodesExplored)

        if problem.goalTest(currNode):
            while currNode in actionRet.keys():
                parent_node = actionRet[currNode]
                solution.insert(0, actionRet[currNode])
                currNode = parent_node
            #print(solution)
            print(actionRet)
            return solution
    
        for i in action:
            #print('i: ', i)
            child_state = problem.getResult(currNode, i)
            #print('Child State: ', child_state)

            if not child_state in nodesExplored:
                #print('i appended: ', i)
                actionRet[problem.getResult(currNode, i)] = currNode
                #print(actionRet)
                child_node = Node(child_state, currNode, i, problem.getCost(child_state, i))
                frontierNodes.push(child_node)'''

    
    util.raiseNotDefined()


def depthFirstSearch(problem):
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def UniformCostSearch(problem):
    """Search the node that has the lowest path cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #get the coordinates of the start state
    start = problem.getStartState()
    #print(start)

    #visited contains nodes that have been removed from the tree and the direction from which they have been obtained
    visited = {}

    #solution contains the sequence of directions for Pacman to get to the goal state
    solution = []

    #node will contain of the co-ordinate values, it's direction, and the cost. start is the root node.
    node = (start, 'Start Node', 0)
    queue = util.PriorityQueue()
    queue.push(node, 0)
    #print(queue)

    #the start state doesn't have a direction from where it's arrived
    visited[start] = 'Start Node'

    #parents will consist of the node and their parents
    parents = {}

    #if start state is goal state
    if problem.goalTest(start):
        return solution

    #stores the cost
    cost = 0

    #stores the dictionary of costs with the cost locations and the cost of each location
    cost_list = {}
    #cost of start node(root node) is 0
    cost_list[start] = cost

    #loop while the goal state has not been reached and the queue is not empty
    goal = False
    while queue != [] and goal != True:
        node = queue.pop()
        #store the element and it's direction
        visited[node[0]] = node[1]
        #if the element is the goal state then exit the loop
        if problem.goalTest(node[0]):
            #print('Worked')
            leaf_node = node[0]
            goal = True
            break
        #else expand the node
        for child in problem.getActions(node[0]):
            #if the child node has not been visited yet
            if problem.getResult(node[0], child) not in visited.keys():
                #create a priority queue
                cost += 1
                priority = node[2] + cost + heuristic(problem.getResult(node[0], child), problem)
                #if new cost is more than old cost, then continue, otherwise push the queue and change cost and parent
                if problem.getResult(node[0], child) in cost_list.keys():
                    if cost_list[problem.getResult(node[0], child)] < priority:
                        continue
                queue.push((problem.getResult(node[0], child), child, node[2] + cost), priority)
                cost_list[problem.getResult(node[0], child)] = priority
                #store the child node and it's parent
                parents[problem.getResult(node[0], child)] = node[0]
                #cost += 1
                #queue.append((problem.getResult(node[0], child), child, cost))
                #print(queue)

    #finding and storing the path
    while(leaf_node in parents.keys()):
        #find the parent
        parent_node = parents[leaf_node]
        #add the direction to the solution
        solution.insert(0, visited[leaf_node])
        #go to the previous node
        leaf_node = parent_node

    return solution


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
