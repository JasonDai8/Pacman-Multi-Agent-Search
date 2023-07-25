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
import random, util, math

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        distList = []
        foodDistList = []
        foodScore = 0.0
        posScore = 0.0
        ghostSpots = [ghost.getPosition() for ghost in newGhostStates]
        foodList = newFood.asList()
        if len(ghostSpots) != 0:
            for ghost in ghostSpots:
                distList.append(math.sqrt((newPos[0] - ghost[0])**2 + (newPos[1] - ghost[1])**2))
            posScore = min(distList)
        if len(foodList) != 0:
            for food in foodList:
                foodDistList.append(math.sqrt((newPos[0] - food[0])**2 + (newPos[1] - food[1])**2))
                foodScore = min(foodDistList)
        return successorGameState.getScore() + posScore - foodScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        v = -99999999
        stop = Directions.STOP
        go = Directions.NORTH
        depth = 0
        holdspot = v
        agentIndex = 1
        for action in actions:
            newGameState = gameState.generateSuccessor(0, action)
            score = self.minimax(newGameState, depth, agentIndex)
            if score != 0:
                holdspot = depth
            if score >= v:
                v = score
                stop = action
        return stop

    def getMax(self, gameState, depth, agentIndex):
        v = -99999999
        nextIndex = 0
        nextDepth = 0
        actions = gameState.getLegalActions(0)
        for action in actions:
            newGameState = gameState.generateSuccessor(0, action)
            each = self.minimax(newGameState, depth, 1)
            v = max(v, each)
        return v
    def getMin(self, gameState, depth, agentIndex):
        v = 99999999
        nextIndex = 0
        nextDepth = 0
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            numAgents = gameState.getNumAgents()
            if agentIndex == numAgents - 1:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                each = self.minimax(newGameState, depth + 1, 0)
                v = min(v, each)
            else:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                each = self.minimax(newGameState, depth, agentIndex + 1)
                v = min(v, each)
        return v
    def minimax(self, gameState, depth, agentIndex):
        nextIndex = 0
        nextDepth = 0
        winState = gameState.isWin()
        loseState = gameState.isLose()
        if (winState or loseState) or (depth == self.depth):
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            findMax = self.getMax(gameState, depth, 0)
            return findMax
        else:
            findMin = self.getMin(gameState, depth, agentIndex)
            return findMin
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        v = -99999999
        alpha = -99999999
        beta = 99999999
        betareset = 0
        stop = Directions.STOP
        go = Directions.NORTH
        depth = 0
        holdspot = v
        agentIndex = 1
        for action in actions:
            newGameState = gameState.generateSuccessor(0, action)
            score = self.minimax(newGameState, depth, agentIndex, alpha, beta)
            if score != 0:
                holdspot = depth
            if score > v:
                v = score
                stop = action
            betareset = beta
            alpha = max(alpha, v)
        return stop

    def getMax(self, gameState, depth, agentIndex, alpha, beta):
        v = -99999999
        nextIndex = 0
        nextDepth = 0
        actions = gameState.getLegalActions(0)
        for action in actions:
            betareset = 0
            newGameState = gameState.generateSuccessor(0, action)
            each = self.minimax(newGameState, depth, 1, alpha, beta)
            v = max(v, each)
            if betareset > depth+1:
                betareset = depth
            if v > beta:
                return v
            betareset = beta
            alpha = max(alpha, v)
        return v
    def getMin(self, gameState, depth, agentIndex, alpha, beta):
        v = 99999999
        nextIndex = 0
        nextDepth = 0
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            betareset = 0
            numAgents = gameState.getNumAgents()
            if agentIndex == numAgents - 1:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                each = self.minimax(newGameState, depth + 1, 0, alpha, beta)
                v = min(v, each)
            else:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                each = self.minimax(newGameState, depth, agentIndex + 1, alpha, beta)
                v = min(v, each)
            if betareset > depth+1:
                betareset = depth
            if v < alpha:
                betareset = beta
                return v
            beta = min(beta, v)
        return v
    def minimax(self, gameState, depth, agentIndex, alpha, beta):
        nextIndex = 0
        nextDepth = 0
        winState = gameState.isWin()
        loseState = gameState.isLose()
        if (winState or loseState) or (depth == self.depth):
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            findMax = self.getMax(gameState, depth, 0, alpha, beta)
            return findMax
        else:
            findMin = self.getMin(gameState, depth, agentIndex, alpha, beta)
            return findMin

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        v = -99999999
        betareset = 0
        holdspot = v
        stop = Directions.STOP
        go = Directions.NORTH
        depth = 0
        agentIndex = 1
        for action in actions:
            newGameState = gameState.generateSuccessor(0, action)
            score = self.minimax(newGameState, depth, agentIndex)
            if score != 0:
                holdspot = depth
            if score >= v:
                v = score
                stop = action
        return stop

    def getMax(self, gameState, depth, agentIndex):
        v = -99999999
        nextIndex = 0
        nextDepth = 0
        actions = gameState.getLegalActions(0)
        for action in actions:
            newGameState = gameState.generateSuccessor(0, action)
            each = self.minimax(newGameState, depth, 1)
            v = max(v, each)
        return v
    def getExp(self, gameState, depth, agentIndex):
        v = 99999999
        hold = 0
        final = 0
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            numAgents = gameState.getNumAgents()
            if agentIndex == numAgents - 1:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                each = self.minimax(newGameState, depth + 1, 0)
                hold = hold + each
            else:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                each = self.minimax(newGameState, depth, agentIndex + 1)
                hold = hold + each
        final = hold/len(actions)
        return final
    def minimax(self, gameState, depth, agentIndex):
        nextIndex = 0
        nextDepth = 0
        winState = gameState.isWin()
        loseState = gameState.isLose()
        if (winState or loseState) or (depth == self.depth):
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            findMax = self.getMax(gameState, depth, 0)
            return findMax
        else:
            findExp = self.getExp(gameState, depth, agentIndex)
            return findExp

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
        this evaluation function is the same as my original evaluation function except now applied to the current game state using the distance formula for both the foodScore and ghostScore
    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    distList = []
    foodDistList = []
    foodScore = 0.0
    posScore = 0.0
    ghostSpots = [ghost.getPosition() for ghost in newGhostStates]
    foodList = newFood.asList()
    if len(ghostSpots) != 0:
        for ghost in ghostSpots:
            distList.append(math.sqrt((newPos[0] - ghost[0])**2 + (newPos[1] - ghost[1])**2))
        posScore = min(distList)
    if len(foodList) != 0:
        for food in foodList:
            foodDistList.append(math.sqrt((newPos[0] - food[0])**2 + (newPos[1] - food[1])**2))
        foodScore = min(foodDistList)
    return currentGameState.getScore() + posScore - foodScore

# Abbreviation
better = betterEvaluationFunction
