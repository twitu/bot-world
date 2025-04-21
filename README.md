# Reinforcement Learning

## Maze Runner

This project implements a value iteration algorithm for a 4x4 GridWorld reinforcement learning problem. The task involves:

- An agent starting at the top-left corner (state 0) trying to reach the bottom-right corner (state 15)
- The agent can move up, down, left, or right with equal probability
- Each move has a reward of -1, and the terminal state has a reward of 0
- No obstacles in the grid

### Implementation

The solution uses value iteration with the Bellman equation to find the optimal value function:

1. Initialize V(s) = 0 for all states
2. Iteratively apply the Bellman equation until convergence:
   ```
   V(s) = R(s) + γ * Σ P(s'|s,a) * V(s')
   ```
   where:
   - R(s) is the immediate reward (-1 for each move)
   - γ (gamma) is the discount factor (set to 1, no discounting)
   - P(s'|s,a) is the transition probability (equal for all valid moves)

3. The algorithm stops when the maximum change in V(s) across all states is < 1e-4

### Running the Code

Execute `python maze_runner.py` to run the value iteration algorithm. The program will display:
- Intermediate value matrices every 10 iterations
- The final value matrix after convergence
- The number of iterations required to converge

### Results

The final value matrix shows the expected total reward (negative of steps) to reach the goal from each position in the grid.

```
[-44.56780443458733, -43.56790073289775, -41.21097701130894, -38.78262595037933]
[-43.56790073289775, -41.925194692437984, -38.282668101043555, -34.35443476401417]
[-41.21097701130894, -38.282668101043555, -31.640381321495838, -22.998226113919294]
[-38.78262595037933, -34.35443476401418, -22.998226113919294, 0]
```

The values might be slighlty different than expected because the solution excludes illegal moves when calculating the expected future rewards.
