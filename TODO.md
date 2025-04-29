Todo

Needed:

-~~ Complete functionality of original gear ratio plot with gear order~~
- ~~Add functionality of plotting all those graphs for different shifting patterns~~
- Add functionality of finding the gearsets that have the best scores, and saving them in a format that can be accepted by graph_shifts()
- Ensure that the name of the gearset actually changes with chainring changing
- Ensure that the combo score comes out as a positive number where 0 is perfect
- Add 11,12,13 to the bottom of every gearset to massively reduce computation time (allows full analysis?)

Nice To Have:
- Add progress bar to track progress of long runs NICETOHAVE
- Work out how to plot large datasets using jupyter NICETOHAVE

If needed:
- add df['your_col'] = df['your_col'].astype('float32') to make smaller in memory for numbers
- add df['your_col'] = df['your_col'].astype('category') to make smaller in memory for text

Complete:
- ~~GET A RECOMMENDATION ON WHICH GEARSET IS BEST!!!~~
- ~~Work out how to cache/ save score results~~
- ~~Include metric on front shifts in score to allow penalisation~~