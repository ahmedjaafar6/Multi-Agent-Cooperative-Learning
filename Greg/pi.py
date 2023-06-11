import numpy as np
import matplotlib.pyplot as plt
from gridworld import *


def polIter(runFor=50000, firstVisit=True, epsilon=0.05, gamma=0.9, averageEvery=100):
  v1 = {}
  v2 = {}
  Rs = {}

  retRewards = []
  retCounts = []
  retMinCounts = []

  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      v1[(r, c)] = 0
      v2[(r, c)] = 0
      Rs[(r, c)] = []

  delta = 0
  for i in range(runFor):

    for r1 in range(worldSize[0]):
      for c1 in range(worldSize[1]):
        s1 = (r1, c1)
        if s1 in walls + [goal]:
          continue
        for r2 in range(worldSize[0]):
          for c2 in range(worldSize[1]):
            s2 = (r2, c2)
            if s2 in walls + [goal]:
              continue

  return v1, v2


def runMC():
  v1, v2, _, _, _ = polIter(
      runFor=10000, firstVisit=True, epsilon=0.2, gamma=0.9)

  piStar1 = getPiStar(v1)
  piStar2 = getPiStar(v2)

  printVpi(v1, piStar1)
  printVpi(v2, piStar2)

  printVpi(v1, piStar1, forLatex=True)
  printVpi(v2, piStar2, forLatex=True)


def runMCwithGraph():
  averageEvery = 10
  v1, v2, rewards, counts, minCounts = polIter(
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
