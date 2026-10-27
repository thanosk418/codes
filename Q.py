import numpy as np
import random

maze = np.array([[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]])
start, goal = (0,0), (3,3)
actions = [(-1,0),(1,0),(0,-1),(0,1)]
Q = np.zeros((16,4))
alpha, gamma, epsilon = 0.1, 0.9, 1.0
rows, cols = maze.shape

def step(s, a):
    r, c = divmod(s, cols)
    dr, dc = actions[a]
    nr, nc = r+dr, c+dc
    if not (0 <= nr < rows and 0 <= nc < cols) or maze[nr,nc]:
        return s, -2, False
    ns = nr*cols+nc
    done = (nr,nc) == goal
    return ns, 10 if done else -1, done

for _ in range(1000):
    s = 0
    for _ in range(50):
        a = random.randrange(4) if random.random() < epsilon else np.argmax(Q[s])
        ns, reward, done = step(s,a)
        target = reward + (gamma*np.max(Q[ns]) if not done else 0)
        Q[s,a] += alpha*(target-Q[s,a])
        s = ns
        if done: break
    epsilon = max(0.05, epsilon*0.995)

s, path, visited = 0, [start], {0}
for _ in range(16):
    s, _, done = step(s, np.argmax(Q[s]))
    if s in visited and not done:
        print("Route loop detected")
        break
    path.append(divmod(s,cols))
    visited.add(s)
    if done:
        print("Goal reached!")
        break

print("Learned path:", path)