import numpy as np

class Grid:
    def __init__(self, board, reward_key, transition_key, random_chance):
        # board is a 2d list of keywords (eg. 'red'). key defines the reward per-keyword
        # random_chance is the probabilty (0.0 - 1.0) of moving wrong
        self.row_size = len(board[0])
        self.col_size = len(board)
        self.rewards = self.reward_table(board, reward_key)                            # a numpy array that defines the reward per-state. int (state) -> float (reward)
        self.transitions = self.transition_table(board, transition_key, random_chance) # a 3d numpy matrix that determines the probability of getting to a state from the current state and action

    def reward_table(self, board, key):
        table = []
        for i in range(self.col_size):
            table.extend([key[board[i][j]] for j in range(self.row_size)])
        return np.array(table)
    
    def transition_table(self, board, key, random_chance):
        # random_chance is the probability (0.0 - 1.0) of going a wrong direction

        table = np.zeros((self.row_size * self.col_size, 4, self.row_size * self.col_size))
        # table[state, action] -> state
        for x in range(self.row_size):
            for y in range(self.col_size):
                start_state = self.loc_to_state((x, y))
                for i, pos in enumerate([(x, y+1), (x+1, y), (x, y-1), (x-1, y)]):
                    probabilities = np.zeros(self.row_size * self.col_size)
                    # list of transition probabilities from the current x, y position
                    sx, sy = pos
                    random_states = [pos]
                    if(x == sx):
                        random_states.extend([(x+1, y), (x-1, y)])
                    else:
                        random_states.extend([(x, y+1), (x, y-1)])
                    
                    for tsx, tsy in random_states:
                        if((tsy < self.col_size) and (tsy >= 0) and (tsx < self.row_size) and (tsx >= 0)):
                            prob = key[board[tsy][tsx]]
                            if(tsx, tsy != sx, sy):
                                prob *= random_chance
                            else:
                                prob -= random_chance * 2
                            probabilities[self.loc_to_state((tsx, tsy))] = prob
                    probabilities[start_state] = 1 - sum(probabilities)

                    table[start_state][i] = probabilities
                               
        return table



    def state_to_loc(self, state):
        return (state % self.row_size, state // self.row_size)
    
    def loc_to_state(self, pos):
        x, y = pos
        return self.row_size * y + x

    def run_policy(self, start, policy):
        # runs a policy step-by-step. should be a list of length self.num_states
        # start is a position
        state = self.loc_to_state(start)
        total_reward = 0
        path = [state]

        while(True):
            # print(self.transitions)
            action = policy[state]
            transition_probabilities = self.transitions[state, action]
            
            if(abs(total_reward) >= 3):
                break
                
            state = np.random.choice(self.row_size * self.col_size, p=transition_probabilities)
            path.append(state)
            total_reward += self.rewards[state]
            if(abs(self.rewards[state]) == 1):
                break
            # print(total_reward)
        return (path, total_reward)
        
g = Grid(
    [
        [1, 1, 1, 2],
        [1, 4, 1, 3],
        [1, 1, 1, 1]
    ],
    {
        1: -0.01,
        2: 1,
        3: -1,
        4: 0
    },
    {
        1: 1,
        2: 1,
        3: 1,
        4: 0
    },
    random_chance=0.2
)



# best_policy = np.random.randint(4, size=15)
# best_average = -1000.0

# for i in range(100):
#     new_policy = np.random.randint(4, size=15)
#     average = 0
#     for _ in range(100):
#         average += g.run_policy((0, 2), new_policy)[1]
#     average /= 1000
#     if(average > best_average):
#         best_policy = new_policy
#         best_average = average
#     print(f"{i+1}/100 complete")

# print(f"Best policy: {best_policy}, average score: {best_average}")
# g.run_policy((2, 0), np.array([
    # 1, 2, 3, 3,
    # 1, 0, 3, 3,
    # 1, 0, 3, 3
# ]))