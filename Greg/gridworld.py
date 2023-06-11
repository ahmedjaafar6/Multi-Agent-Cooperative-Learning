import random
import numpy as np

worldSize = (6, 5)
goal = (5, 4)
initDoor = (5, 2)
door = list(initDoor)
button = (2, 2)
water = (0, 2)
walls = [(1, 2), (3, 2), (2, 3), (4, 2)]
# walls = walls + [(3, 0), (3, 1)]

# probabilities = {'go': 0.8, 'turnleft': 0.05, 'turnright': 0.05, 'stay': 0.1}
probabilities = {'go': 1, 'turnleft': 0, 'turnright': 0, 'stay': 0}

agent1_start = 2, 0
agent2_start = 5, 0

otherAgent = [0, 0]

actionMap = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1),
             'right': (0, 1), 'none': (0, 0)}
actions = ['up', 'right', 'down', 'left']

gamma = 0.8


def resetDoor():
  door[0] = initDoor[0]
  door[1] = initDoor[1]


def openDoor():
  door[0] = -1
  door[1] = -1


def getPiStar(v):
  piStar = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      if (r, c) in walls:
        continue
      piStar[(r, c)] = pi((r, c), v, epsilon=0)
  return piStar


def printVpi(V, piStar, forLatex=False):
  print('V:')
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      print('{0:.4f}'.format(V.get((r, c), 0)), end=' ')
      if forLatex and c < worldSize[1] - 1:
        print(' & ', end='')
      if forLatex and c == worldSize[1] - 1:
        print('\\\\', end='')

    print()

  print('pi:')
  print('  ', end='')
  for c in range(worldSize[1]):
    print(c, end=' ')
  print()
  for r in range(worldSize[0]):
    if not forLatex:
      print(r, end=' ')
    for c in range(worldSize[1]):
      if (r, c) == goal:
        print('G', end=' ')
      elif (r, c) in walls:
        print(' ' if forLatex else '#', end=' ')
      elif piStar.get((r, c), None) is None:
        print('?', end=' ')
      elif piStar.get((r, c), None) == 'up':
        print('\\uparrow' if forLatex else '↑', end=' ')
      elif piStar.get((r, c), None) == 'down':
        print('\\downarrow' if forLatex else '↓', end=' ')
      elif piStar.get((r, c), None) == 'left':
        print('\\leftarrow' if forLatex else '←', end=' ')
      elif piStar.get((r, c), None) == 'right':
        print('\\rightarrow' if forLatex else '→', end=' ')
      if forLatex and c < worldSize[1] - 1:
        print(' & ', end='')
      if forLatex and c == worldSize[1] - 1:
        print('\\\\', end='')
    print()


def R(s, ns, qLearning=False):
  if ns == goal:
    return 10
  elif ns == water:
    if qLearning:
      return -0.05
    return 0
    # return -10
  else:
    if qLearning:
      return -0.05
    return 0


def move(state, attemptedAction):
  r, c = state
  actionIndex = actions.index(attemptedAction)
  randVal = random.random()
  if randVal < probabilities['go']:
    action = attemptedAction
  elif randVal < probabilities['go'] + probabilities['stay']:
    action = 'none'
  elif randVal < probabilities['go'] + probabilities['stay'] + probabilities['turnleft']:
    action = actions[(actionIndex + 1) % 4]
  else:
    action = actions[(actionIndex - 1) % 4]

  dr, dc = actionMap[action]

  if r + dr < 0 or r + dr >= worldSize[0] or c + dc < 0 or c + dc >= worldSize[1]:
    newState = state
  elif (r + dr, c + dc) in walls + [tuple(door), tuple(otherAgent)]:
    newState = state
  else:
    newState = (r + dr, c + dc)

  if newState == button:
    openDoor()

  if state == button and newState != button:
    resetDoor()

  return newState


# p = {}
# for r in range(worldSize[0]):
#   for c in range(worldSize[1]):
#     state = (r, c)
#     for action in actions:
#       actionIndex = actions.index(action)
#       ns = tuple(np.add(state, actionMap[action]))
#       probStay = probabilities['stay']
#       if ns[0] < 0 or ns[0] >= worldSize[0] or ns[1] < 0 or ns[1] >= worldSize[1] or ns in walls + [tuple(door), tuple(otherAgent)]:
#         probStay += probabilities['go']
#       else:
#         p[state, action, ns] = probabilities['go']

#       ns = tuple(np.add(state, actionMap[actions[(actionIndex + 1) % 4]]))
#       if ns[0] < 0 or ns[0] >= worldSize[0] or ns[1] < 0 or ns[1] >= worldSize[1] or ns in walls + [tuple(door), tuple(otherAgent)]:
#         probStay += probabilities['turnleft']
#       else:
#         p[state, action, ns] = probabilities['turnleft']

#       ns = tuple(np.add(state, actionMap[actions[(actionIndex - 1) % 4]]))
#       if ns[0] < 0 or ns[0] >= worldSize[0] or ns[1] < 0 or ns[1] >= worldSize[1] or ns in walls + [tuple(door), tuple(otherAgent)]:
#         probStay += probabilities['turnright']
#       else:
#         p[state, action, ns] = probabilities['turnright']

#       p[state, action, state] = probStay


def pFunc(s, a, ns):
  # if dist > 1 return 0.0
  dist = abs(s[0] - ns[0]) + abs(s[1] - ns[1])
  if dist > 1:
    return 0.0

  probStay = probabilities['stay']

  # fix if gonna do stochastic
  predNs = tuple(np.add(s, actionMap[a]))
  if predNs in walls or predNs[0] < 0 or predNs[0] >= worldSize[0] or predNs[1] < 0 or predNs[1] >= worldSize[1]:
    return 1.0 if ns == s else 0.0
  return 1.0 if ns == predNs else 0.0

  if predNs == ns:
    return probabilities['go']

  # predictedNext = tuple(np.add(s, actionMap[a]))
  # if predictedNext == ns and predictedNext not in walls and predictedNext[0] >= 0 and predictedNext[0] < worldSize[0] and predictedNext[1] >= 0 and predictedNext[1] < worldSize[1]:
  #   return probabilities['go']
  # else:
  #   probStay += probabilities['go']

  # predictedNext = tuple(np.add(s, actionMap[actions[(actions.index(a) + 1) % 4]]))
  # if predictedNext == ns and predictedNext not in walls and predictedNext[0] >= 0 and predictedNext[0] < worldSize[0] and predictedNext[1] >= 0 and predictedNext[1] < worldSize[1]:
  #   return probabilities['turnleft']
  # else:
  #   probStay += probabilities['turnleft']

  # predictedNext = tuple(np.add(s, actionMap[actions[(actions.index(a) - 1) % 4]]))
  # if predictedNext == ns and predictedNext not in walls and predictedNext[0] >= 0 and predictedNext[0] < worldSize[0] and predictedNext[1] >= 0 and predictedNext[1] < worldSize[1]:
  #   return probabilities['turnright']
  # else:
  #   probStay += probabilities['turnright']

  # predictedNext = tuple(np.add(s, actionMap[actions[(actions.index(a) + 2) % 4]]))
  # if predictedNext == ns and predictedNext not in walls and predictedNext[0] >= 0 and predictedNext[0] < worldSize[0] and predictedNext[1] >= 0 and predictedNext[1] < worldSize[1]:
  #   return 0.0

  # return probStay


def q(s, a, v):
  # if s == button:
  #   print('pause')
  nsSum = 0
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      ns = (r, c)
      if ns not in walls:
        if ns == goal and pFunc(s, a, ns) > 0:
          pass
        if s == button:
          pass
        # if s == button and p.get((s, a, ns), 0) > 0:
        if s == button and pFunc(s, a, ns) > 0:
          pass
        # nsSum += p.get((s, a, ns), 0) * (R(s, ns) + gamma * v.get(ns, 0))
        nsSum += pFunc(s, a, ns) * (R(s, ns) + gamma * v.get(ns, 0))
  return nsSum


def pi(s, v, epsilon):
  if s == (4, 4) or s == (5, 3):
    pass
  if s == (2, 2):
    pass
  bestActions = []
  bestActionValue = float('-inf')
  for action in actions:
    actionValue = q(s, action, v)
    if actionValue > bestActionValue:
      bestActions = [action]
      bestActionValue = actionValue
    elif actionValue == bestActionValue:
      bestActions.append(action)

  actionProbs = {}
  for action in actions:
    if action in bestActions:
      actionProbs[action] = (1 - epsilon) / \
          len(bestActions) + epsilon / len(actions)
    else:
      actionProbs[action] = epsilon / len(actions)

  randVal = random.random()
  for action in actions:
    if randVal < actionProbs[action]:
      return action
    else:
      randVal -= actionProbs[action]


def piQ(s, q, epsilon, actions=actions):
  bestActions = []
  bestActionValue = float('-inf')
  for action in actions:
    actionValue = q.get((s, action), 0)
    if actionValue > bestActionValue:
      bestActions = [action]
      bestActionValue = actionValue
    elif actionValue == bestActionValue:
      bestActions.append(action)

  actionProbs = {}
  for action in actions:
    if action in bestActions:
      actionProbs[action] = (1 - epsilon) / \
          len(bestActions) + epsilon / len(actions)
    else:
      actionProbs[action] = epsilon / len(actions)

  randVal = random.random()
  for action in actions:
    if randVal < actionProbs[action]:
      return action
    else:
      randVal -= actionProbs[action]
