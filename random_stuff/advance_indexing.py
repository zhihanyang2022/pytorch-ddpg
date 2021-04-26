import numpy as np

row = np.arange(0, 10).reshape(1, 10)

episodes = np.vstack([row] * 5)

print('Before indexing:')
print(episodes)

row_indices = []
col_indices = []

row_to_col = {
    0: 3,
    1: 4,
    2: 5,
    3: 0,
    4: 1
}

for row_index in range(5):
    for col_index in range(row_to_col[row_index], row_to_col[row_index]+3):
        row_indices.append(row_index)
        col_indices.append(col_index)

print('After indexing:')
print(episodes[row_indices, col_indices].reshape(-1, 3))