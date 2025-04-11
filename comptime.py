from main import best_cadence
from main import score

for i in range(1):
    no_in_rear = 5+i
    largest_rear = 16+i
    sprocket_params=(no_in_rear,2,11,largest_rear,34,50)
    chainring_params=(50,34)
    best_cadence()
    score("quarters")
