import os

try:
    os.makedirs('data/ones')
    os.makedirs('data/alternation')
except:
    pass

with open('data/ones/input.txt', 'w') as f:
    for i in range(10000):
        f.write('1')
    f.flush()


with open('data/alternation/input.txt', 'w') as f:
    for i in range(10000):
        f.write(str(i%2))
    f.flush()
