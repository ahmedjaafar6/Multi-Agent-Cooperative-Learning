import numpy as np
import matplotlib.pyplot as plt
from gridworld import *


def QLearning(runFor=50000, firstVisit=True, epsilon=0.05, gamma=0.9, alpha=0.5, averageEvery=100):
  # v1 = {}
  # v2 = {}

  Q1 = {}
  Q2 = {}

  Rs = {}

  retRewards = []
  retCounts = []
  retMinCounts = []

  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      for a in actions:
        Q1[((r, c), a)] = 0
        Q2[((r, c), a)] = 0

      Rs[(r, c)] = []

  endRewards = []
  endSteps = []
  for j in range(runFor):
    if j % 1000 == 0:
      print('episode', j)
      pass

    initRandom = False
    if initRandom:
      s1 = (np.random.randint(
          0, worldSize[0]), np.random.randint(0, worldSize[1]))
      while s1 in walls + [tuple(door), tuple(otherAgent)]:
        s1 = (np.random.randint(
            0, worldSize[0]), np.random.randint(0, worldSize[1]))

      s2 = (np.random.randint(
          0, worldSize[0]), np.random.randint(0, worldSize[1]))
      while s2 in walls + [tuple(door), tuple(otherAgent)]:
        s2 = (np.random.randint(
            0, worldSize[0]), np.random.randint(0, worldSize[1]))
    else:
      s1, s2 = agent1_start, agent2_start

    resetDoor()

    episode1, episode2 = [], []
    rewards = []
    count = 0
    while s1 != goal and s2 != goal:

      a1 = piQ(s1, Q1, epsilon=epsilon)
      ns1 = move(s1, a1)
      episode1.append(s1)

      otherAgent[0] = ns1[0]
      otherAgent[1] = ns1[1]

      a2 = piQ(s2, Q2, epsilon=epsilon)
      ns2 = move(s2, a2)
      episode2.append(s2)

      otherAgent[0] = ns2[0]
      otherAgent[1] = ns2[1]

      rsum = R(s1, ns1, qLearning=False) + R(s2, ns2, qLearning=False)

      rewards.append(rsum)

      Q1[(s1, a1)] = Q1.get((s1, a1), 0) + alpha * \
          (rsum + gamma * max([Q1.get((ns1, a), 0) for a in actions]) -
           Q1.get((s1, a1), 0))

      Q2[(s2, a2)] = Q2.get((s2, a2), 0) + alpha * \
          (rsum + gamma * max([Q2.get((ns2, a), 0) for a in actions]) -
           Q2.get((s2, a2), 0))

      s1 = ns1
      s2 = ns2

      count += 1

    endRewards.append(sum(rewards))
    endSteps.append(count)

    if j % averageEvery == 0:
      retRewards.append(np.mean(endRewards))
      retCounts.append(np.mean(endSteps))
      retMinCounts.append(np.min(endSteps + retMinCounts))
      endRewards = []
      endSteps = []

    v1Arr = np.zeros(worldSize)
    v1Obj = {}
    for r in range(worldSize[0]):
      for c in range(worldSize[1]):
        v1Arr[r, c] = np.max(
            [Q1.get(((r, c), a), 0) for a in actions])
        v1Obj[(r, c)] = v1Arr[r, c]

    v2Arr = np.zeros(worldSize)
    v2Obj = {}
    for r in range(worldSize[0]):
      for c in range(worldSize[1]):
        v2Arr[r, c] = np.max(
            [Q2.get(((r, c), a), 0) for a in actions])
        v2Obj[(r, c)] = v2Arr[r, c]

  return Q1, Q2, v1Obj, v2Obj, retRewards, retCounts, retMinCounts


def runQL():
  Q1, Q2, v1, v2, _, _, _ = QLearning(
      runFor=10000, firstVisit=True, epsilon=0.2, gamma=0.9, alpha=0.1)

  piHat1 = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      if (r, c) in walls:
        continue
      piHat1[(r, c)] = piQ((r, c), Q1, epsilon=0)

  piHat2 = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      if (r, c) in walls:
        continue
      piHat2[(r, c)] = piQ((r, c), Q2, epsilon=0)

  printVpi(v1, piHat1)
  printVpi(v2, piHat2)

  printVpi(v1, piHat1, forLatex=True)
  printVpi(v2, piHat2, forLatex=True)


def runQLwithGraph():
  averageEvery = 10
  Q1, Q2, v1, v2, rewards, counts, minCounts = QLearning(
      runFor=10000, firstVisit=True, epsilon=0.2, gamma=gamma, alpha=0.2, averageEvery=averageEvery)

  piHat1 = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      if (r, c) in walls:
        continue
      piHat1[(r, c)] = piQ((r, c), Q1, epsilon=0)

  piHat2 = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      if (r, c) in walls:
        continue
      piHat2[(r, c)] = piQ((r, c), Q2, epsilon=0)

  printVpi(v1, piHat1)
  printVpi(v2, piHat2)

  printVpi(v1, piHat1, forLatex=True)
  printVpi(v2, piHat2, forLatex=True)

  # make graph with left y axis for rewards and right y axis for counts
  fig, ax = plt.subplots()
  ax2 = ax.twinx()

  # ax1.plot(rewards, 'b-')
  ax.plot(counts, 'r-')
  ax2.plot(minCounts, 'g-')

  ax.set_xlabel('Episode (x' + str(averageEvery) + ')')
  # ax1.set_ylabel('Average Reward', color='b')
  ax.set_ylabel('Average Steps', color='r')
  ax2.set_ylabel('Min Steps', color='g')

  # set plot y to only show 0 to 100 steps
  ax.set_ylim([0, 50])
  ax2.set_ylim([0, 50])

  plt.show()


if __name__ == '__main__':
  # runQL()
  runQLwithGraph()
