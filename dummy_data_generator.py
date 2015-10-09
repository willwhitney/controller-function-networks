import os

try:
    os.makedirs('data/ones')
except:
    pass
try:
    os.makedirs('data/alternation')
except:
    pass
try:
    os.makedirs('data/double_alternation')
except:
    pass

# with open('data/ones/input.txt', 'w') as f:
#     for i in range(10000):
#         f.write('1')
#     f.flush()
#
#
# with open('data/alternation/input.txt', 'w') as f:
#     for i in range(10000):
#         f.write(str(i%2))
#     f.flush()

with open('data/double_alternation/input.txt', 'w') as f:
    for i in range(10000):
        if i % 4 < 2:
            f.write("0")
        else:
            f.write("1")
    f.flush()
