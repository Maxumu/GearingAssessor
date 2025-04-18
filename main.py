import pandas as pd
import numpy as np
# from fontTools.varLib.varStore import storeancer
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from itertools import combinations
import csv
import time

from config import GearConfig
from config import VarStore

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

def sprocket_generator(config: GearConfig):
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
    # Imports arguments from config file
    possible_sprockets = list(range(config.smallest_rear, config.largest_rear + 1))
    cassette_combinations = list(combinations(possible_sprockets, config.max_rear))

    # Generates options for rear sprockets

    cassette_options = {}
    for cassette in cassette_combinations:
        key = "-".join(map(str, cassette))
        cassette_options[key] = (cassette)
    global number_generated
    number_generated = len(cassette_combinations)
    print(f"sprocket_generator done, generated: {number_generated}")

    return(cassette_options)

def gearset_generator(config: GearConfig, chainring_params =[50,34]):

    cassette_options = sprocket_generator(config)
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

def calculate_ratios(config: GearConfig):
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
    generated_gearsets = gearset_generator(config)

    if config.use_real and config.use_generated:
        gear_combinations = pd.concat([real_gear_combinations, generated_gearsets])

    elif not config.use_real and config.use_generated:
        gear_combinations = generated_gearsets

    elif config.use_real and not config.use_generated:
        gear_combinations = real_gear_combinations

    elif not config.use_real and not config.use_generated:
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
    long_real = calculate_ratios(config.use_real,config.use_generated)
    unique = long_real['RearTeeth'].unique()
    print(np.sort(unique))
    return

def drivetrain_splitter(config: GearConfig):
    """
    Input:
        Long dataframe with all gears and ratios
    Returns:
        Dictionary of drivetrain dataframes, each one named by 'cassete_chainring'
    """
    gear_combinations = calculate_ratios(config)

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

def shifting_pattern(config: GearConfig, pattern):
    """Sorts the gears into a shifting pattern based on the selected input

    Input:
        Which pattern to use (Ideal,Halves,Quarters)
    Output:
        Gear combinations flat dataframe
        """

    drivetrains = drivetrain_splitter(config)

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


def calculate_jumps(config: GearConfig, pattern):
    """
    Input:
        Pandas dataframe formatted database for each single combination
    Returns:
        Jumps for each gear and adds them to each single combination
    """
    drivetrains = shifting_pattern(config,pattern)

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
    fit_A = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]
    store = VarStore(fit_A = fit_A, fit_B = fit_B, fit_C = fit_C)

    # Plots fitted quadratic curve on same axis as datapoints and saves png to file
    # fit_efficiency = quadratic(high_res_cadence, fit_A, fit_B, fit_C)
    # plt.plot(high_res_cadence, fit_efficiency, "-")
    # plt.savefig("plot.png")

    # print("cadence_reference done")
    return(store)

def best_cadence(store: VarStore):
    """
    Input:
        Derivative of fitted cadence/efficiency curve to find optimal cadence
    Returns:
        Best cadence in RPM
    """
    # Set derivative of func = 0 to find maximum
    # store = cadence_reference()
    peak_cadence = (-store.fit_B) / (2 * store.fit_A)
    store.peak_cadence = peak_cadence

    # print("best_cadence done")
    return(store)

def efficiency(cadence_in,store: VarStore):
    """
    Input:
        Cadence in rpm
    Returns:
        Normalised efficiency as proportion of maximum
    """

    fit_A = store.fit_A
    fit_B = store.fit_B
    fit_C = store.fit_C

    peak_cadence = store.peak_cadence
    peak_efficiency = (fit_A*(peak_cadence**2) + fit_B*(peak_cadence) + fit_C)

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

def score(config: GearConfig,store: VarStore,pattern):
    """
    Input:
        Groupset dataframe with ratio jumps
    Returns single long dataframe with groupset id column:
        Optimal gear jump
        Worst gear jump
        Avg gear jump - optimal gear jump
        """
    all_scores = []
    groupsets = calculate_jumps(config,pattern)

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
        effective_cadence = (worst_jump_diff / optimal_jump) * store.peak_cadence
        worst_efficiency = efficiency(effective_cadence,store)

        # Calculates average jump across whole groupset, compares to optimal
        average_jump = df["Jump_avg"].mean()
        avg_to_opt = abs(average_jump - optimal_jump)

        all_scores.append({
            "Groupset": key,
            "Optimal Jump": optimal_jump,
            "Worst Diff to Optimal": worst_jump_diff,
            "Avg Diff to Optimal": avg_to_opt,
            "Average jump": average_jump,
            "Range": range,
            "Worst Efficiency": worst_efficiency,
            "Biggest Jump": biggest_jump,
            "Smallest Jump": smallest_jump,
            "Effective Cadence": effective_cadence
        })
        # Adds dataframe for each groupest to dictionary of scores
        all_scores_df = pd.DataFrame(all_scores)

    # SHIM11_30TDF = scores_dict['11-30 Shimano 12_TDF Pro']

    print("score done")
    return(all_scores_df)

def results_plotter_matplotlib(all_scores_df):

    plt.figure(figsize=(10, 10))
    leader = 1
    for key, group in all_scores_df.groupby("Groupset"):
        # Extract the values for plotting
        x_value = group["Worst Efficiency"]
        y_value = group["Range"]
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

def results_plotter_fast():
    """
    Input:
    Scores database for all
    Returns:

    """

# def input_parameters(realin,generatedin,no_in_rear,largest_rear):
#     real = realin
#     generated = generatedin
#     sprocket_params = (no_in_rear, 2, 11, largest_rear, 34, 50)
#     chainring_params = (50, 34)
#     score("quarters")
#     return

def main():
    """
    Initialises config file available to all functions
    """
    config = GearConfig()

if __name__ == "__main__":
    print("Running")
    main()
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

    time_dict = {}
    for i in range(5):
        start = time.time()
        chainring_params = (50, 34)
        store = cadence_reference()
        store = best_cadence(store)
        config = GearConfig(use_real=False,use_generated=True,max_rear=6,largest_rear=16+i)
        results_plotter_matplotlib(score(config,store,"quarters"))
        end = time.time()
        duration = end - start
        key = number_generated
        time_dict[key] = duration

    with open("time_dict1.csv", "w") as csv_file:
        gearnum = time_dict.keys()
        times = time_dict.values()
        stats = zip(gearnum, times)
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerows(stats)
        # writer.writerow(time_dict.values())
    print(time_dict)
        # input_parameters(False, True, 6, largest_rear)