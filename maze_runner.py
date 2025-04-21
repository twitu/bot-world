import random


class Maze:
    def __init__(self, size):
        self.size = size
        self.maze = [[0 for _ in range(size)] for _ in range(size)]
        self.start = (0, 0)
        self.end = (size - 1, size - 1)

    def is_valid_position(self, position):
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def valid_neighbors(self, position):
        moves = [
            (position[0] + 1, position[1]),
            (position[0] - 1, position[1]),
            (position[0], position[1] + 1),
            (position[0], position[1] - 1),
        ]
        return [move for move in moves if self.is_valid_position(move)]


class Policy:
    def __init__(self, maze):
        self.maze = maze
        self.value_matrix = [[0 for _ in range(maze.size)] for _ in range(maze.size)]
        """convergence threshold"""
        self.theta = 0.0001
        """discount factor for future rewards"""
        self.gamma = 1

    def print_value_matrix(self):
        for i in range(self.maze.size):
            print(self.value_matrix[i])
            
    def value_iteration(self):
        """Perform one sweep of value iteration using the Bellman equation"""
        max_change = 0
        
        # Create a copy of the current value matrix
        new_value_matrix = [row[:] for row in self.value_matrix]
        
        # For each state
        for i in range(self.maze.size):
            for j in range(self.maze.size):
                # Skip the terminal state
                if (i, j) == self.maze.end:
                    continue
                
                current_pos = (i, j)
                # Get all valid next states
                next_states = self.maze.valid_neighbors(current_pos)
                
                # Calculate new value using Bellman equation
                # V(s) = -1 + gamma * sum[P(s'|s,a) * V(s')]
                # Since all actions are equally likely, P(s'|s,a) = 1/len(next_states)
                new_value = -1  # Immediate reward for taking a step
                
                # Add expected future rewards
                for next_pos in next_states:
                    transition_prob = 1.0 / len(next_states)
                    new_value += self.gamma * transition_prob * self.value_matrix[next_pos[0]][next_pos[1]]
                
                # Update the value
                new_value_matrix[i][j] = new_value
                
                # Track maximum change
                max_change = max(max_change, abs(new_value - self.value_matrix[i][j]))
        
        # Update the value matrix
        self.value_matrix = new_value_matrix
        
        return max_change


class MazeRunner:
    def __init__(self, maze):
        self.maze = maze
        self.position = (0, 0)

    def run(self):
        positions = [self.position]

        while self.position != self.maze.end:
            moves = self.maze.valid_neighbors(self.position)
            if not moves:
                return False
            self.position = random.choice(moves)
            positions.append(self.position)

        return positions

    def reset(self):
        self.position = self.maze.start


if __name__ == "__main__":
    maze = Maze(4)
    policy = Policy(maze)
    max_change = 1
    iter = 0

    while max_change > policy.theta:
        max_change = policy.value_iteration()
        iter += 1

        if iter % 10 == 0:
            print(f"Iteration {iter}")
            print(f"Max change: {max_change}")
            print("Value matrix:")
            policy.print_value_matrix()
            print()

    print(f"Converged in {iter} iterations")
    print(f"Max change: {max_change}")
    print("Final value matrix:")
    policy.print_value_matrix()
