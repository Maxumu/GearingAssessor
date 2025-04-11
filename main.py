from os.path import realpath
import pandas as pd
import numpy as np
from numpy.f2py.symbolic import normalize
from pandas.conftest import indexer_al
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import hypothesis
import pytest
from itertools import combinations
from datetime import datetime

def csv_2_dataframe(csv_file):
    """
    Input:
        Takes in csv data of cassette descriptions
    Returns:
        Dataframe with gear ratios calculated
    """
    read_csv = pd.read_csv(csv_file)
    # print(read_csv.to_string())
    read_rear_string = read_csv.head()
    # print("csv_2_dataframe done")
    return read_csv
    

def sprocket_generator(max_rear = 6, max_front = 2, smallest_rear = 11, largest_rear = 22, smallest_front = 34, largest_front = 54):
    """
    Input:
        Number of sprockets, number of chainrings
    Output:
        Long dataframe with every possible gearset
        Columns:
            "Cassette" (ID for rear gears)
            "RearTeeth" (Tooth number for rear gear)
            "Chainring" (ID for chainring)
            "FrontTeeth" (Tooth number for front gear)

    Generates every possible gearset for a given number of sprockets and chainrings.
    Limits on number of teeth:
        Smallest rear sprocket = 11
        Largest rear sprocket = 36
        Smallest front chainring = 34
        Largest front chainring = 50
    Cassette structure:
        Cassete must be in ascending order of size with no duplicates
        Chainring must be in decending order of size with no duplicates
        Rear sprocket number = 12
        Front chainring number = 2
    """

    # Generates options for rear sprockets

    possible_sprockets = list(range(smallest_rear, largest_rear + 1))
    cassette_combinations = list(combinations(possible_sprockets, max_rear))

    cassette_options = {}
    for cassette in cassette_combinations:
        key = "-".join(map(str, cassette))
        cassette_options[key] = (cassette)

    number_generated = len(cassette_combinations)
    print(f"sprocket_generator done, generated: {number_generated}")

    return(cassette_options)

def gearset_generator(sprocket_params,chainring_params=[50,34]):

    cassette_options = sprocket_generator(*sprocket_params)
    data = []
    chainring_descriptor = "-".join(map(str,chainring_params))

    for cassette_key, sprockets in cassette_options.items():
        for rear_teeth in sprockets:
            for front_teeth in chainring_params:
                data.append({
                    "Cassette": cassette_key,
                    "RearTeeth": rear_teeth,
                    "Chainring": chainring_descriptor,
                    "FrontTeeth": front_teeth
                })
    generated_gearsets = pd.DataFrame(data)
    print("gearset_generator done")
    return(generated_gearsets)

def calculate_ratios(real,generated):
    """
    Input:
        CSV files of gear ratios from front and rear
    Returns:
        Long dataframe with ratios between rear and front
    """

    # Reads dataframes and puts in a flat dataframe
    read_rear = csv_2_dataframe('gearing_database.csv')
    read_front = csv_2_dataframe('front_gears.csv')
    flat_rear = read_rear.melt(var_name="Cassette", value_name="RearTeeth")
    flat_front = read_front.melt(var_name="Chainring", value_name="FrontTeeth")
    real_gear_combinations = flat_rear.merge(flat_front, how="cross")

    # Makes generated flat dataframe
    generated_gearsets = gearset_generator(sprocket_params, chainring_params)

    if real and generated:
        gear_combinations = pd.concat([real_gear_combinations, generated_gearsets])

    elif not real and generated:
        gear_combinations = generated_gearsets

    elif real and not generated:
        gear_combinations = real_gear_combinations

    elif not real and not generated:
        print("No gears to analyse")
        return

    # Calculates gear ratios
    gear_combinations["GearRatio"] = gear_combinations["FrontTeeth"] / gear_combinations["RearTeeth"]

    # # Calls function that sorts for shifting order
    # gear_combinations = shifting_pattern(pattern)
    #
    # # Pulls out single combinations and stores them in a big dictionary
    # single_combinations = {}
    # for (cassette,chainring), groupset in gear_combinations.groupby(["Cassette","Chainring"]):
    #     key = f"{cassette}_{chainring}"
    #     single_combinations[key] = groupset.drop(columns=["Cassette","Chainring"])
    #
    # # Exports whole big dictionary to excel spreadsheet with dataframes in separate sheets
    # with pd.ExcelWriter("/home/m-hahn-ubuntu/Documents/GEE401/python_gear_ratios.xlsx", engine="xlsxwriter") as writer:
    #     for sheet_name, df in single_combinations.items():
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)
    #
    # # Converts to html table for easy viewing
    # html_combinations = gear_combinations.to_html()
    # text_file = open("html_combinations.html","w")
    # text_file.write(html_combinations)
    # text_file.close()
    print("calculate_ratios done")
    return(gear_combinations)

def unique_sprockets():
    """
    Input:
        Long dataframe with all real gears
    Output:
        Unique rear sprocket tooth numbers
    """
    long_real = calculate_ratios(True,False)
    unique = long_real['RearTeeth'].unique()
    print(np.sort(unique))
    return

def drivetrain_splitter():
    """
    Input:
        Long dataframe with all gears and ratios
    Returns:
        Dictionary of drivetrain dataframes, each one named by 'cassete_chainring'
    """
    gear_combinations = calculate_ratios(real,generated)

    # Splits long dataframe into individual drivetrains and stores in dictionary
    drivetrains = {}
    for (cassette, chainring), groupset in gear_combinations.groupby(["Cassette", "Chainring"]):
        key = f"{cassette}_{chainring}"
        drivetrains[key] = groupset.drop(columns=["Cassette", "Chainring"])

    # # Exports whole big dictionary to excel spreadsheet with dataframes in separate sheets
    # with pd.ExcelWriter("/home/m-hahn-ubuntu/Documents/GEE401/python_gear_ratios.xlsx", engine="xlsxwriter") as writer:
    #     for sheet_name, df in drivetrains.items():
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Converts to html table for easy viewing
    html_combinations = gear_combinations.to_html()
    text_file = open("html_combinations.html", "w")
    text_file.write(html_combinations)
    text_file.close()
    print("drivetrain_splitter done")
    return(drivetrains)

def shifting_pattern(pattern):
    """Sorts the gears into a shifting pattern based on the selected input

    Input:
        Which pattern to use (Ideal,Halves,Quarters)
    Output:
        Gear combinations flat dataframe
        """

    drivetrains = drivetrain_splitter()

    # Chooses the pattern to sort the dataframe
    for key,drivetrain in drivetrains.items():
        if pattern == "ideal":
            drivetrains[key] = drivetrain.sort_values(by=("GearRatio"),ascending=[True])
        elif pattern == "halves":
            drivetrains[key] = drivetrain.sort_values(by=["FrontTeeth","RearTeeth"],ascending=[True, False])
        elif pattern == "quarters":
            # Pre sorts the dataframe into halves pattern to make simpler 2nd operation
            drivetrain = drivetrain.sort_values(by=["FrontTeeth","RearTeeth"],ascending=[True, False])
            # drivetrain = drivetrains[key]
            combos = int(drivetrain.shape[0])
            remainder = combos % 4
            print(f"combos: {combos}")
            # Gear combos multiple of 4
            if remainder == 0:
                first = drivetrain.iloc[0:(combos//4)]
                second = drivetrain.iloc[(combos//4):(2*combos//4)]
                third = drivetrain.iloc[(2*combos//4):(3*combos//4)]
                fourth = drivetrain.iloc[3*combos//4:]

            # Indivisible gears = 1, distributes 1st quadrant
            if remainder == 1:
                first_end = combos // 4 + 1
                first = drivetrain.iloc[0:first_end]
                second_end = first_end + (combos // 4)
                second = drivetrain.iloc[first_end:second_end]
                third_end = second_end + (combos // 4)
                third = drivetrain.iloc[second_end:third_end]
                fourth = drivetrain.iloc[third_end:]

            # Indivisible gears = 2, distributes 1st, 3rd quadrants
            if remainder == 2:
                first_end = combos//4 + 1
                first = drivetrain.iloc[0:first_end]
                second_end = first_end + (combos//4)
                second = drivetrain.iloc[first_end:second_end]
                third_end = second_end + (combos//4) + 1
                third = drivetrain.iloc[second_end:third_end]
                fourth = drivetrain.iloc[third_end:]

            # Indivisible gears = 3, distributes 1st, 2nd, 3rd quadrants
            if remainder == 3:
                first_end = combos // 4 + 1
                first = drivetrain.iloc[0:first_end]
                second_end = first_end + (combos // 4) + 1
                second = drivetrain.iloc[first_end:second_end]
                third_end = second_end + (combos // 4) + 1
                third = drivetrain.iloc[second_end:third_end]
                fourth = drivetrain.iloc[third_end:]

            # Resorts quarters into order
            drivetrains[key] = pd.concat([first, third, second, fourth], ignore_index=True)

        else:
            print("Not an option buddy")
            return
    
    print("shifting_pattern done")
    return(drivetrains)


def calculate_jumps(pattern):
    """
    Input:
        Pandas dataframe formatted database for each single combination
    Returns:
        Jumps for each gear and adds them to each single combination
    """
    drivetrains = shifting_pattern(pattern)

    for key, df in drivetrains.items():
        df["Jump_up"] = abs(1-(df["GearRatio"] / df["GearRatio"].shift(-1)))
        df["Jump_down"] = abs(1-(df["GearRatio"] / df["GearRatio"].shift(1)))
        df["Jump_avg"] = df[["Jump_up","Jump_down"]].mean(axis=1)
        drivetrains[key] = df

    # Exports whole big dictionary to excel spreadsheet with dataframes in separate sheets
    # with pd.ExcelWriter("/home/m-hahn-ubuntu/Documents/GEE401/python_gear_ratios_jumps.xlsx", engine="xlsxwriter") as writer:
    #     for sheet_name, df in drivetrains.items():
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)
    # SHIM11_30TDF = drivetrains['11-30 Shimano 12_TDF Pro']
    print("calculate_jumps done")
    return(drivetrains)

# def graphit():
#     """Takes the data on each of the groupsets with jumps calculated
#         Outputs a graph of jumps against each physical gear combination"""
#     groupsets = calculate_jumps()
#     for df in groupsets.items():
#         plt.plot((df["GearRatio"]),(df["Jump_up"]))
#     return

def quadratic(x,a,b,c):
    """
    Defines quadratic function for curve fit
    """
    quad = (a*x**2)+(b*x)+c
    # print("quadratic done")
    return(quad)

def cadence_reference():
    """
    Input:
        Cadence in rpm
        Uses csv file of cadence vs gross efficiency data to fit quadratic curve
    Returns:
        Fits curve and exports png to file
        Returns efficiency for given cadence
    """
    # Inputs data from csv file and plots on axis
    cadence_data = csv_2_dataframe("cadence_vs_efficiency.csv")
    cadence = cadence_data["Cadence"]
    efficiency = cadence_data["Gross Efficiency"]
    plt.plot(cadence,efficiency,"o")

    # Interpolate cadence to make fitted curve smoother on graph
    cad_range = max(cadence) - min(cadence)
    high_res_cadence = []
    for i in range(21):
        high_res_cadence.append(min(cadence)+((i/20)*cad_range))
    high_res_cadence = pd.Series(high_res_cadence)

    # Uses curve_fit to fit quadratic function as defined to the cadence and efficiency
    parameters,covarience = curve_fit(quadratic,cadence,efficiency)
    global fit_A
    global fit_B
    global fit_C
    fit_A = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]

    # Plots fitted quadratic curve on same axis as datapoints and saves png to file
    # fit_efficiency = quadratic(high_res_cadence, fit_A, fit_B, fit_C)
    # plt.plot(high_res_cadence, fit_efficiency, "-")
    # plt.savefig("plot.png")

    # print("cadence_reference done")
    return(parameters)

def best_cadence():
    """
    Input:
        Derivative of fitted cadence/efficiency curve to find optimal cadence
    Returns:
        Best cadence in RPM
    """
    # Set derivative of func = 0 to find maximum
    parameters = cadence_reference()
    fit_A = parameters[0]
    fit_B = parameters[1]
    global peak_cadence
    peak_cadence = (-fit_B) / (2 * fit_A)

    # print("best_cadence done")
    return()

def efficiency(cadence_in):
    """
    Input:
        Cadence in rpm
    Returns:
        Normalised efficiency as proportion of maximum
    """
    best_cad = peak_cadence
    peak_efficiency = (fit_A*(best_cad**2) + fit_B*(best_cad) + fit_C)

    # Calculates efficiency for input cadence
    efficiency_out = quadratic(cadence_in,fit_A,fit_B,fit_C)
    # print(f"Efficiency is: ", float(efficiency_out))

    # Normalise curve for use with cost function:
        # We want to compensate for the fact that lots of the gross efficiencies we can't control, the maximum
        # efficiency is like 20% So that maximum should become = 1, and proportional differences either side.
    y_normal_quad_func = ((fit_A*(cadence_in**2) + fit_B*(cadence_in) + fit_C)) / peak_efficiency
    # print(f"Normalised efficiency: ",y_normal_quad_func)
    plt.plot()

    """Current optimal cadence is ~60 rpm, and what I want ideally is the multiplier away from that centre point, as that is
     what I can calculate based on the difference from optimal ratio. This requires a parameter 'proportional difference
     from optimal gear ratio', so that it can be multiplied by the optimal cadence.
         """
    # print("efficiency done")
    return(y_normal_quad_func)

def score(pattern):
    """
    Input:
        Groupset dataframe with ratio jumps
    Returns in dict with groupset key:
        Optimal gear jump
        Worst gear jump
        Avg gear jump - optimal gear jump
        FUTURE: efficiency from bio data
        """
    scores_dict = {}
    groupsets = calculate_jumps(pattern)

    for key, df in groupsets.items():
        score = []

        # Calculates optimal shift for even shifting
        range = df["GearRatio"].max() / df["GearRatio"].min()
        optimal_jump = (range**(1/((df.shape[0])-1)))-1

        # Calculates worst gear changes
        biggest_jump = np.nanmax(df[["Jump_up","Jump_down"]].to_numpy())
        smallest_jump = np.nanmax(df[["Jump_up","Jump_down"]].to_numpy())
        worst_jump_diff_big = biggest_jump - optimal_jump
        worst_jump_diff_small = smallest_jump - optimal_jump
        worst_jump_diff = max(worst_jump_diff_small,worst_jump_diff_big)

        # Finds proportional difference from optimal cadence, applies it to efficiency func for worst efficiency shift
        best_cad = peak_cadence
        effective_cadence = (worst_jump_diff / optimal_jump)*best_cad
        worst_efficiency = efficiency(effective_cadence)

        # Calculates average jump across whole groupset, compares to optimal
        average_jump = df["Jump_avg"].mean()
        avg_to_opt = abs(average_jump - optimal_jump)

        # Builds list of scores
        score.append([optimal_jump,worst_jump_diff,avg_to_opt,average_jump,range,worst_efficiency,biggest_jump,smallest_jump,effective_cadence])

        # Converts list of scores to dataframe
        scores = pd.DataFrame(score, columns = ["Optimal Jump","Worst Diff to Optimal","Avg Diff to Optimal","Average jump","Range","Worst Efficiency","Biggest Jump","Smallest Jump","Effective Cadence"])

        # Adds dataframe for each groupest to dictionary of scores
        scores_dict[key] = scores

    # SHIM11_30TDF = scores_dict['11-30 Shimano 12_TDF Pro']

    print("score done")
    return(scores_dict)

def results_plotter(score_dict):
    plt.figure(figsize=(10, 10))
    leader = 1
    for key, df in score_dict.items():
        # Extract the values for plotting
        x_value = df["Worst Efficiency"]
        y_value = df["Range"]
        plt.scatter(x_value, y_value, label=key)
        # plt.text(x_value, y_value, key, fontsize=6, ha='right')
        if x_value.iloc[0] < leader:
            leader = x_value.iloc[0]
            leader_key = key
    
    plt.xlabel("Worst Efficiency")
    plt.ylabel("Range")
    plt.title("Score DataFrames Scatter Plot")
    plt.grid(True)
    # plt.legend()
    # plt.show()
    plt.savefig("plot2.png")
    print("results_plotter done")
    return

def input_parameters(real,generated,no_in_rear,largest_rear):
    return([real,generated,no_in_rear,largest_rear])

if __name__ == "__main__":
    print("Running")
    real = input_parameters()[0]
    generated = input_parameters()[1]
    no_in_rear = input_parameters()[2]
    largest_rear = input_parameters()[3]

    sprocket_params = (no_in_rear, 2, 11, largest_rear, 34, 50)
    chainring_params = (50, 34)
    best_cadence()
    score("quarters")


    # start = datetime.now()
    # print(f"Start time: {start}")
    # # score("ideal")
    # results_plotter(score("ideal"))
    # end = datetime.now()
    # print(f"End time: {end}")
    # duration = end - start
    # print(f"Duration: {duration}")
    # shifting_pattern("quarters")
    # unique_sprockets()