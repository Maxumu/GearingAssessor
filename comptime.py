from main import input_parameters
from main import best_cadence
from main import score

for i in range(1):
    no_in_rear = 5+i
    largest_rear = 16+i
    input_parameters(False,True,no_in_rear,largest_rear)
    best_cadence()
    score("quarters")