from gridworld import *

def td(alpha=0.5, epsilon=0.05):
  v = {}

  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      v[(r, c)] = 0

  count = 0
  while True:
    agent1 = (np.random.randint(
        0, worldSize[0]), np.random.randint(0, worldSize[1]))
    while agent1 in walls + [tuple(door), tuple(otherAgent)]:
      agent1 = (np.random.randint(
          0, worldSize[0]), np.random.randint(0, worldSize[1]))

    agent2 = (np.random.randint(
        0, worldSize[0]), np.random.randint(0, worldSize[1]))
    while agent2 in walls + [tuple(door), tuple(otherAgent)]:
      agent2 = (np.random.randint(
          0, worldSize[0]), np.random.randint(0, worldSize[1]))

    while agent1 != goal and agent2 != goal:
      action = pi(agent1, v, epsilon=epsilon)
      nextAgent1 = move(agent1, action)

      v[agent1] = v[agent1] + alpha * \
          (R(agent1, nextAgent1) + gamma * v[nextAgent1] - v[agent1])

      agent1 = nextAgent1

    vArr = np.zeros(worldSize)
    for r in range(worldSize[0]):
      for c in range(worldSize[1]):
        vArr[r, c] = v.get((r, c), 0)

    if count >= 10000:
      break

    count += 1
  return v


def one():
  vs, epNums = [], []
  for i in range(1):
    v, epNum = td(alpha=0.03, epsilon=0.0)
    vs.append(v)
    epNums.append(epNum)

  v = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      v[(r, c)] = np.mean([v[(r, c)] for v in vs])

  piHat = {}
  for r in range(worldSize[0]):
    for c in range(worldSize[1]):
      if (r, c) in walls:
        continue
      piHat[(r, c)] = pi((r, c), v, epsilon=0)
  printVpi(v, piHat)
  printVpi(v, piHat, forLatex=True)

  print('avg epNum', np.mean(epNums))
  print('std dev epNum', np.std(epNums))


if __name__ == '__main__':
  one()