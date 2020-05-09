# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
        "*** YOUR CODE HERE **"
        closestghost = min([manhattanDistance(newPos,ghost.getPosition()) for ghost in newGhostStates])

        if closestghost:
            ghost_dist = -10/closestghost
        else:
            ghost_dist = -1000

        foodlist = newFood.asList()
        if foodlist:
           food_dist = min([manhattanDistance(newPos,Foodpos) for Foodpos in foodlist])
        else:
           food_dist = 0

        return (-2*food_dist) + ghost_dist - 100*len(foodlist) + successorGameState.getScore()
        
        #return successorGameState.getScore()
 
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def MinimaxSearch(state,agentIndex,depth):
            if agentIndex == state.getNumAgents():
                if depth == self.depth: return self.evaluationFunction(state)
                else: return MinimaxSearch(state,0,depth + 1)

            else: 
                moves = state.getLegalActions(agentIndex)

                if len(moves) == 0: return self.evaluationFunction(state)

                evaluation_successor = (MinimaxSearch(state.generateSuccessor(agentIndex, direct),agentIndex + 1,depth) for direct in moves)    

                if agentIndex == 0: return max(evaluation_successor)
                return min(evaluation_successor)
        
        result = max(gameState.getLegalActions(0), key=lambda x: MinimaxSearch(gameState.generateSuccessor(0,x),1,1))
        
        return result
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabetaprunning(state):
            value, bestAction = None, None
            alpha, beta = None, None

            for action in state.getLegalActions(0):
                value = max(value,minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta))
                if alpha is None:
                    alpha = value
                    bestAction = action
                else:
                    alpha, bestAction = max(value, alpha), action if value > alpha else bestAction
            return bestAction

        def minValue(state, agentIdx, depth, alpha, beta):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1, alpha, beta)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)
                if alpha is not None and value < alpha:
                    return value
                if beta is None:
                    beta = value
                else:
                    beta = min(beta, value)
            
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state)

            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                value = max(value, succ)
                if beta is not None and value > beta:
                    return value
                alpha = max(alpha, value)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabetaprunning(gameState)

        return action
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax_search(state, agentIndex, depth):
            # if in min layer and last ghost
            if agentIndex == state.getNumAgents():
                # if reached max depth, evaluate state
                if depth == self.depth:
                    return self.evaluationFunction(state)
                # otherwise start new max layer with bigger depth
                else:
                    return expectimax_search(state, 0, depth + 1)
            # if not min layer and last ghost
            else:
                moves = state.getLegalActions(agentIndex)
                # if nothing can be done, evaluate the state
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                # get all the minimax values for the next layer with each node being a possible state after a move
                next = (expectimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

                # if max layer, return max of layer below
                if agentIndex == 0:
                    return max(next)
                # if min layer, return expectimax values
                else:
                    l = list(next)
                    return sum(l) / len(l)
        # select the action with the greatest minimax value
        result = max(gameState.getLegalActions(0), key=lambda x: expectimax_search(gameState.generateSuccessor(0, x), 1, 1))

        return result

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacmanPos = currentGameState.getPacmanPosition()
    FoodPos = currentGameState.getFood().asList()
    Capsulespos = currentGameState.getCapsules()
    GhostStates = currentGameState.getGhostStates()
    
    FoodDis = [manhattanDistance(pacmanPos,PFood) for PFood in FoodPos]
    CapsulesDis = [manhattanDistance(pacmanPos,Capsules) for Capsules in Capsulespos]

    score = 0

    maxdisGhost = float("-inf")
    mindisGhost = float("inf")
    for ghostState in GhostStates:
        dis = manhattanDistance(pacmanPos,ghostState.getPosition())
        if dis < 2:
            if ghostState.scaredTimer != 0:
                score += 10/(dis + 1)
            else: 
                score -= 10/(dis + 1)
        maxdisGhost = max(maxdisGhost, dis)
        mindisGhost = min(mindisGhost, dis)

    
    numFood = len(FoodDis)
    if numFood > 0:
        score += (50/numFood)
    #print(pacmanPos)
    
    numCapules = len(CapsulesDis)
    if numCapules > 0:
        score -= (5/numCapules)
    else : score += 40

    score -= 0.003*mindisGhost

    MindisFood = min(FoodDis + [10])
    MinCapsule = min(CapsulesDis + [10])
    return currentGameState.getScore() + score - 0.2*MindisFood + 0.3*MinCapsule  + 100/(MinCapsule + 0.03*MindisFood + 0.3*maxdisGhost);
    
    util.raiseNotDefined()
     

# Abbreviation
better = betterEvaluationFunction

