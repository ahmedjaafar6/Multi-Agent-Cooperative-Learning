import numpy as np
import matplotlib.pyplot as plt
from gridworld import *


def monteCarlo(runFor=50000, firstVisit=True, epsilon=0.05, gamma=0.9, averageEvery=100):
  v1 = {}
  v2 = {}
  Rs = {}

  retRewards = []
  retCounts = []
  retMinCounts = []

  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      # 10 - 0.1 * manhattan distance from goal
      # if not (r, c) in walls + [tuple(door), tuple(otherAgent), goal]:
      #   dist = 1 - 0.1 * (abs(r - goal[0]) + abs(c - goal[1]))
      #   v1[(r, c)] = dist
      #   v2[(r, c)] = dist

      v1[(r, c)] = 0
      v2[(r, c)] = 0
      Rs[(r, c)] = []

  endRewards = []
  endSteps = []
  for j in range(runFor):
    if j % 100 == 0:
      print('episode', j)
      # piStar = {}
      # for r in range(worldSize[0]):
      #   for c in range(worldSize[1]):
      #     if (r, c) in walls + [tuple(door), tuple(otherAgent)]:
      #       continue
      #     piStar[(r, c)] = pi((r, c), v1, epsilon=0)
      # printVpi(v, piStar)
      # printVpi(v, piStar, forLatex=True)
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
      # action = pi(state, v, epsilon=epsilon)
      # nextState = move(state, action)
      # episode.append(state)
      # rewards.append(R(state, nextState))
      # state = nextState

      a1 = pi(s1, v1, epsilon=epsilon)
      ns1 = move(s1, a1)
      episode1.append(s1)

      otherAgent[0] = ns1[0]
      otherAgent[1] = ns1[1]

      a2 = pi(s2, v2, epsilon=epsilon)
      ns2 = move(s2, a2)
      episode2.append(s2)

      otherAgent[0] = ns2[0]
      otherAgent[1] = ns2[1]

      rewards.append(R(s1, ns1) + R(s2, ns2))

      s1 = ns1
      s2 = ns2

      if (s1 == (4, 2) or s2 == (4, 2)):
        pass

      if (s1 == button or s2 == button):
        pass

      count += 1

    endRewards.append(sum(rewards))
    endSteps.append(count)

    G = 0
    # iterate backwards through episode
    for i in range(len(episode1) - 1, -1, -1):
      s1, s2 = episode1[i], episode2[i]

      G = gamma * G + rewards[i]
      if firstVisit:
        if s1 not in episode1[:i]:
          Rs[s1].append(G)
          v1[s1] = np.mean(Rs[s1])
        if s2 not in episode2[:i]:
          Rs[s2].append(G)
          v2[s2] = np.mean(Rs[s2])
      else:
        Rs[s1].append(G)
        v1[s1] = np.mean(Rs[s1])
        Rs[s2].append(G)
        v2[s2] = np.mean(Rs[s2])

    if j % averageEvery == 0:
      retRewards.append(np.mean(endRewards))
      retCounts.append(np.mean(endSteps))
      retMinCounts.append(np.min(endSteps + retMinCounts))
      endRewards = []
      endSteps = []

  return v1, v2, retRewards, retCounts, retMinCounts


def runMC():
  v1, v2, _, _, _ = monteCarlo(
      runFor=10000, firstVisit=True, epsilon=0.2, gamma=0.9)

  piStar1 = getPiStar(v1)
  piStar2 = getPiStar(v2)

  printVpi(v1, piStar1)
  printVpi(v2, piStar2)

  printVpi(v1, piStar1, forLatex=True)
  printVpi(v2, piStar2, forLatex=True)


def runMCwithGraph():
  averageEvery = 10
  v1, v2, rewards, counts, minCounts = monteCarlo(
      runFor=10000, firstVisit=True, epsilon=0.05, gamma=gamma, averageEvery=averageEvery)

  piStar1 = getPiStar(v1)
  piStar2 = getPiStar(v2)

  printVpi(v1, piStar1)
  printVpi(v2, piStar2)

  printVpi(v1, piStar1, forLatex=True)
  printVpi(v2, piStar2, forLatex=True)

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
  # runMC()
  runMCwithGraph()
