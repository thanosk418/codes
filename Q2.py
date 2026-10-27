# %load https://raw.githubusercontent.com/antigravity112/d/main/Q
import numpy as np
import random

# 0 = free cell, 1 = wall
maze = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0]
])

start, goal = (0, 0), (7, 7)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

rows, cols = maze.shape
Q = np.zeros((64, 4))

alpha, gamma, epsilon = 0.1, 0.9, 1.0

def step(state, action):
    r, c = divmod(state, cols)
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc

    if (nr < 0 or nr >= rows or nc < 0 or nc >= cols
            or maze[nr, nc] == 1):
        return state, -2, False

    next_state = nr * cols + nc
    done = (nr, nc) == goal
    reward = 10 if done else -1

    return next_state, reward, done

# Training
for episode in range(2000):
    state = 0

    for _ in range(100):
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        target = reward
        if not done:
            target += gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        state = next_state

        if done:
            break

    epsilon = max(0.05, epsilon * 0.995)

# Find the learned path
state = 0
path = [start]
visited = {state}

for _ in range(64):
    action = np.argmax(Q[state])
    next_state, reward, done = step(state, action)

    if next_state == state or (next_state in visited and not done):
        print("Route loop detected. Try training longer.")
        break

    state = next_state
    position = divmod(state, cols)
    path.append(position)
    visited.add(state)

    if done:
        print("Goal reached!")
        break
else:
    print("Goal not reached. Try training longer.")

print("Learned path:", path)

# Display the maze
for r in range(rows):
    line = ""
    for c in range(cols):
        if (r, c) == start:
            line += " S "
        elif (r, c) == goal:
            line += " G "
        elif maze[r, c] == 1:
            line += " # "
        elif (r, c) in path:
            line += " * "
        else:
            line += " . "
    print(line)
