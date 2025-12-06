from swings import find_swings  # DİKKAT: swings (sonunda S var)

prices = [1, 2, 3, 4, 5, 4, 3, 2, 1,
          2, 3, 2, 1,
          2, 3, 4, 3, 2, 1]

swings = find_swings(prices, window=2, min_distance=2)

for sp in swings:
    print(sp)
