import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
from itertools import combinations
import csv
import time
import os
import glob
from tqdm import tqdm

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
    read_rear_string = read_csv.head()
    print("csv_2_dataframe done")
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
        Smallest front chainring = 40
        Largest front chainring = 54
    Cassette structure:
        Cassette must be in ascending order of size with no duplicates
        Chainring must be in decending order of size with no duplicates
        Rear sprocket number = 12
        Front chainring number = 2
    """
    # Imports arguments from config file

    # Define locked size sprockets
    bottom_locked = []
    bottom_locked = [
        config.smallest_rear,
        config.smallest_rear + 1,
        config.smallest_rear + 2
    ]
    top_locked = config.largest_rear

    # Check enough free slots
    num_free = config.max_rear - (len(bottom_locked) + 1)
    if num_free < 0:
        raise ValueError("max_rear must be at least 4 to accommodate 3 bottom + 1 top.")

    # Candidates for the “middle” sprockets run from just above bottom_locked up to below top_locked
    middle_candidates = range(bottom_locked[-1] + 1, top_locked)
    middle_combos = combinations(middle_candidates, num_free)

    # Generates options for rear sprockets
    cassette_options = {}
    for combo in middle_combos:
        cassette = tuple(bottom_locked) + tuple(combo) + (top_locked,)
        key = "-".join(map(str, cassette))
        cassette_options[key] = cassette
    global number_generated
    number_generated = len(cassette_options)
    print(f"sprocket_generator done, generated: {number_generated}")

    return(cassette_options)

def gearset_generator(config: GearConfig):

    cassette_options = sprocket_generator(config)
    chainrings = [config.largest_front,config.smallest_front]
    data = []
    chainring_descriptor = "-".join(map(str,(chainrings)))

    for cassette_key, sprockets in cassette_options.items():
        for rear_teeth in sprockets:
            for front_teeth in chainrings:
                data.append({
                    "Cassette": cassette_key,
                    "RearTeeth": rear_teeth,
                    "Chainring": chainring_descriptor,
                    "FrontTeeth": front_teeth
                })
    generated_gearsets = pd.DataFrame(data)
    # print(generated_gearsets.head)
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

    # Loads generated flat dataframe
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

def unique_sprockets(config: GearConfig):
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
    # html_combinations = gear_combinations.to_html()
    # text_file = open("html_combinations.html", "w")
    # text_file.write(html_combinations)
    # text_file.close()
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
        elif pattern == "halves_all_combos":
            drivetrain = drivetrain.sort_values(by=["FrontTeeth","RearTeeth"],ascending=[True, False])
            drivetrains[key] = drivetrain
        elif pattern == "halves_half_combos":
            # Pre-sort the dataframe by front teeth (ascending) and rear teeth (descending)
            drivetrain = drivetrain.sort_values(by=["FrontTeeth","RearTeeth"],ascending=[True, False])
            
            # Get unique front teeth values
            front_teeth_values = drivetrain['FrontTeeth'].unique()
            
            # Initialize empty list to store dataframes for each half
            half_dfs = []
            
            # For each front sprocket, create a separate dataframe
            for front_teeth in front_teeth_values:
                front_df = drivetrain[drivetrain['FrontTeeth'] == front_teeth]
                
                # Sort by rear teeth (descending) to ensure proper order
                front_df = front_df.sort_values(by="RearTeeth", ascending=False)
                
                # Calculate how many rear sprockets to use for this front sprocket
                rear_teeth_values = front_df['RearTeeth'].unique()
                sprockets_per_front = len(rear_teeth_values) // len(front_teeth_values)
                
                # Handle remaining sprockets (distribute remainder starting from first front sprocket)
                remainder_idx = list(front_teeth_values).index(front_teeth)
                if remainder_idx < len(rear_teeth_values) % len(front_teeth_values):
                    sprockets_per_front += 1
                
                # Select the appropriate rear sprockets for this front sprocket
                start_idx = 0
                for i in range(remainder_idx):
                    extra = 1 if i < len(rear_teeth_values) % len(front_teeth_values) else 0
                    start_idx += (len(rear_teeth_values) // len(front_teeth_values)) + extra
                
                end_idx = start_idx + sprockets_per_front
                selected_rears = rear_teeth_values[start_idx:end_idx]
                
                # Add selected combinations to the result
                selected_df = front_df[front_df['RearTeeth'].isin(selected_rears)]
                half_dfs.append(selected_df)
            
            # Combine all half dataframes and ensure the final result is in ascending order
            combined_df = pd.concat(half_dfs, ignore_index=True)
            drivetrains[key] = combined_df.sort_values(by="GearRatio", ascending=True)
        elif pattern == "quarters":
            # Pre sorts the dataframe into half pattern to make simpler 2nd operation
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
        Dictionary of pandas dataframes, each one with patterned gearset
    
    Calculates jumps for each patterned gearset and adds to dataframes in dictionary.
    
    Returns:
        Same dictionary of dataframes
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
        
    # Save each dataframe to its own Parquet file
    output_dir = "parquet_drivetrains"
    os.makedirs(output_dir, exist_ok=True)
    for file in glob.glob(os.path.join(output_dir, "*.parquet")):
        os.remove(file)
    
    start_parq = time.time()
    print(f"starting writing drivetrains to parquet: {start_parq}")
    
    # for key, df in drivetrains.items():
    #     filename = os.path.join(output_dir, f"{key.replace(' ', '_').replace('/', '-')}.parquet")
    #     df.to_parquet(filename, index=False)

    combined_df = pd.concat(
        [df.assign(dataset_name=key) for key, df in drivetrains.items()],
        ignore_index=True
    )

    combined_df.to_parquet("drivetrains_combined.parquet", index=False)

    end_parq = time.time()
    duration_parq = end_parq - start_parq
    print(f"Duration (s): {duration_parq}")

    print("calculate_jumps done")
    return(drivetrains)

def graph_shifts(gearsets_to_plot):
    """
    Takes the data on each of the groupsets with jumps calculated
    Outputs a graph of jumps against each physical gear combination
    """
    gearsets = gearsets_to_plot

    output_dir = "gear_plots"
    os.makedirs(output_dir, exist_ok=True)

    for file in glob.glob(os.path.join(output_dir, "*.png")):
        os.remove(file)

    for name, df in gearsets.items():
        # Map FrontTeeth to ascending index (smallest = 1, etc.)
        front_order = {v: i+1 for i, v in enumerate(sorted(df['FrontTeeth'].unique()))}

        # Map RearTeeth to descending index (largest = 1, etc.)
        rear_order = {v: i+1 for i, v in enumerate(sorted(df['RearTeeth'].unique(), reverse=True))}

        # Assign index columns
        df['FrontIndex'] = df['FrontTeeth'].map(front_order)
        df['RearIndex'] = df['RearTeeth'].map(rear_order)

        # Create GearCombo in the form rear_front (e.g., 1_2)
        df['GearCombo'] = df['RearIndex'].astype(str) + "_" + df['FrontIndex'].astype(str)

        plt.figure(figsize=(12, 6))
        plt.scatter(df['GearCombo'], df['GearRatio'], color='black', zorder=3)

        # Draw colored lines between gear shifts
        for i in range(len(df) - 1):
            x1, x2 = df['GearCombo'].iloc[i], df['GearCombo'].iloc[i + 1]
            y1, y2 = df['GearRatio'].iloc[i], df['GearRatio'].iloc[i + 1]
            rear1, rear2 = df['RearIndex'].iloc[i], df['RearIndex'].iloc[i + 1]
            front1, front2 = df['FrontIndex'].iloc[i], df['FrontIndex'].iloc[i + 1]

            if rear1 != rear2 and front1 == front2:
                color = 'blue'   # Rear shift
            elif rear1 == rear2 and front1 != front2:
                color = 'yellow'    # Front shift
            else:
                color = 'red'   # Both shifted

            plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)

        # Ideal gear ratio curve
        G_high = df['GearRatio'].max()
        G_low = df['GearRatio'].min()

        R_T = G_high / G_low
        R_ideal = R_T**(1 / (len(df) - 1))
        ideal_ratios = [G_low * (R_ideal ** i) for i in range(len(df))]
        plt.plot(range(len(df)), ideal_ratios, color="lightgrey", label='Ideal Gear Ratios (6.77% jumps)', marker='o', zorder=1)

        # Labeling and styling
        plt.xticks(rotation=90, fontsize=12)  # x-axis ticks
        plt.yticks(fontsize=12)               # y-axis ticks
        plt.xlabel('Gear Combination', fontsize=14)
        plt.ylabel('Gear Ratio', fontsize=14)
        plt.title(f"Shifting Pattern Gear Ratios by Gear Combination {name}", fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        # Legend
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Rear shift'),
            Line2D([0], [0], color='yellow', lw=2, label='Front shift'),
            Line2D([0], [0], color='red', lw=2, label='Both shifted'),
            Line2D([0], [0], color='lightgrey', lw=2, label='Ideal shift')
        ]
        plt.legend(handles=legend_elements, title="Shift Type", fontsize=12, title_fontsize=13)

        plt.savefig(f"{output_dir}/{name}_gear_ratio.png")
        plt.close()
    return

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
    # plt.figure(figsize=(10, 6))
    # plt.plot(cadence,efficiency,"o",label="Data(check)")

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
    stnd_dev_err = np.sqrt(np.diag(covarience))
    # print(f"Coefficients: {parameters}")
    # print(f"1 Standard Dev curve_fit error: {stnd_dev_err}")

    # Stores coefficients for future calculations
    store = VarStore(fit_A = fit_A, fit_B = fit_B, fit_C = fit_C)

    J = np.vstack([high_res_cadence**2,high_res_cadence,np.ones_like(high_res_cadence)]).T
    y_fit = quadratic(high_res_cadence, *parameters)
    y_err = np.sqrt(np.sum((J @ covarience) * J, axis=1))  # 1-sigma error

    # plt.plot(high_res_cadence, y_fit, "-", label="Fitted Curve")
    # plt.fill_between(high_res_cadence, y_fit - y_err, y_fit + y_err, color='gray', alpha=0.3, label="±1σ Confidence Band")


    # # Plots fitted quadratic curve on same axis as datapoints and saves png to file
    # fit_efficiency = quadratic(high_res_cadence, fit_A, fit_B, fit_C)
    # plt.plot(high_res_cadence, fit_efficiency, "-")
    # plt.savefig("plot.png")

    # Format equation for legend
    equation_label = (f"Fitted Curve: "f"$y = {fit_A:.4f}x^2 + {fit_B:.4f}x + {fit_C:.4f}$")

       # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cadence, efficiency, "o", label="Measured Data", color="black")
    plt.plot(high_res_cadence, y_fit, "-", label=equation_label, color="green")

    # Labels and grid
    plt.xlabel("Cadence (RPM)")
    plt.ylabel("Gross Efficiency")
    plt.title("Cadence vs Gross Efficiency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Legend", loc='best')
    plt.tight_layout()

    # Save and/or show
    plt.savefig("cadence_efficiency_plot.png")
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
        smallest_jump = np.nanmin(df[["Jump_up","Jump_down"]].to_numpy())
        worst_jump_diff_big = biggest_jump - optimal_jump 
        worst_jump_diff_small = optimal_jump - smallest_jump # changed this so that it is a positive value 
        worst_jump_diff = max(worst_jump_diff_small,worst_jump_diff_big)

        # Finds proportional difference from optimal cadence, applies it to efficiency func for worst efficiency shift
        worst_effective_cadence = ((worst_jump_diff / optimal_jump)+1) * store.peak_cadence
        worst_efficiency = efficiency(worst_effective_cadence,store)
        worst_efficiency_formatted = 1-worst_efficiency

        # Calculates average jump across whole groupset, compares to optimal, applies efficiency function
        df["up_differences_from_opt"] = df["Jump_up"]/optimal_jump
        rms_diff = np.sqrt(np.mean((df["up_differences_from_opt"])**2))
        rms_effective_cadence = rms_diff * store.peak_cadence
        rms_eff = efficiency(rms_effective_cadence,store)
        rms_eff_formatted = 1-rms_eff

        # average_jump = df["Jump_avg"].mean()
        # avg_to_opt = abs(average_jump - optimal_jump)
        # avg_to_opt_proportional = avg_to_opt / optimal_jump
        # avg_effective_cadence = avg_to_opt_proportional * store.peak_cadence
        # avg_efficiency = efficiency(avg_effective_cadence,store)
        # avg_efficiency_formatted = 1-avg_efficiency
    
        # Calculates number of front shifts
        front_changes = (df["FrontTeeth"] != df["FrontTeeth"].shift()).sum() -1

        # Calculates single combo_score from all measures
        combo_score = worst_efficiency_formatted + rms_eff_formatted + front_changes

        all_scores.append({
            "Groupset": key,
            "Range": range,
            "Optimal Jump": optimal_jump,
            "RMS diff": rms_diff,
            "RMS cad": rms_effective_cadence,
            "RMS Efficiency": rms_eff,
            "RMS eff formatted": rms_eff_formatted,
            # "Average Effective Cadence": avg_effective_cadence,
            "Worst Diff to Optimal": worst_jump_diff,
            # "Avg Diff to Optimal": avg_to_opt,
            # "Average jump": average_jump,
            # "Biggest Jump": biggest_jump,
            # "Smallest Jump": smallest_jump,
            # "Average Efficiency": avg_efficiency,
            "Worst Effective Cadence": worst_effective_cadence,
            "Worst Efficiency": worst_efficiency,
            "Worst Efficiency Formatted": worst_efficiency_formatted,
            # "Average Efficiency Formatted": avg_efficiency_formatted,
            "Front Shifts": front_changes,
            "Combo Score": combo_score
        })

    # Converts to dataframe
    all_scores_df = pd.DataFrame(all_scores)
    print(f"Rows in score: {all_scores_df.shape[0]}")

    # SHIM11_30TDF = scores_dict['11-30 Shimano 12_TDF Pro']
    print("score done")
    return(all_scores_df)

def get_data(config: GearConfig,store: VarStore,pattern,force_recompute=False):
    '''
    If parquet already saved and not forced recompute return dataframe
    Else runs score and saves the output as a parquet file, then returns the dataframe
    '''
    CACHE_FILE = "bigdata.parquet"

    if os.path.exists(CACHE_FILE) and not force_recompute:
        print("loading cached data")
        all_scores_df = pd.read_parquet(CACHE_FILE)
    else:
        print("creating cached data")
        all_scores_df = score(config,store,pattern)
        all_scores_df.to_parquet(CACHE_FILE)
    return(all_scores_df)

def best_finder(config: GearConfig, store: VarStore, pattern):
    """
    Takes in scores dataframe and decides which gearsets are best, then seperates them
    """
    all_scores_df = get_data(config,store,pattern)

    top_rows = []
    # Defines how much of the score dataframe to hold in ram at once
    chunk_size = 500_000
    # Defines what proportion of the score dataframe I want to keep to plot
    top_size = 60
    # Goes through dataframe in chunks
    for start in range(0,len(all_scores_df),chunk_size):
        chunk = all_scores_df.iloc[start:start+chunk_size]
        top = chunk.nsmallest(top_size,"Combo Score")
        top_rows.append(top)

    all_top_rows = pd.concat(top_rows)
    top_of_top = all_top_rows.nsmallest(top_size,"Combo Score")

    top_of_top.to_csv('top_of_top.csv', index=False)
    top_of_top.to_parquet('top_of_top.parquet')

    print("best_finder done")
    return(top_of_top)

def results_plotter_matplotlib(all_scores_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    texts = []

    for key, group in all_scores_df.groupby("Groupset"):
        label = key[:-8]
        x_value = group["Worst Efficiency"]
        y_value = group["RMS Efficiency"]
        ax.scatter(x_value, y_value, label=label, zorder=3)

        for x, y in zip(x_value, y_value):
            text = ax.annotate(
                label,
                xy=(x, y),
                xytext=(x, y),
                textcoords='data',
                fontsize=12,
                zorder=2,
                bbox=dict(facecolor='white', alpha=1, edgecolor='none'))
            texts.append(text)


    fig.canvas.draw()

    adjust_text(
        texts,
        ax=ax,
        # arrowprops=dict(arrowstyle='->', color='grey', lw=1, shrinkA=2, shrinkB=1),
        force_points=2,
        force_text=1,
        expand_points=(1, 1),
        expand_text=(1, 1),
        only_move={'points':'xy', 'text': 'xy'},
        precision=0.01,
        zorder=1
    )

    ax.set_xlabel("Worst Efficiency", fontsize=14)
    ax.set_ylabel("RMS Efficiency", fontsize=14)
    ax.set_title("Score DataFrames Scatter Plot", fontsize=14)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig("Worst_RMS_Efficiency.png")
    print("results_plotter done")


    return



def time_predict():
    """
    Predicts time for large datasets
    """
    time_dict = {}
    for i in range(1):
        start = time.time()
        store = cadence_reference()
        store = best_cadence(store)
        config = GearConfig(use_real=False, use_generated=True, max_rear=11, largest_rear=23 + i)
        results_plotter_matplotlib(score(config, store, "quarters"))
        end = time.time()
        duration = end - start
        key = number_generated
        time_dict[key] = duration

    # Saves times from each iteration to csv file to predict large dataset time

    with open("time_dict1.csv", "w") as csv_file:
        gearnum = time_dict.keys()
        times = time_dict.values()
        stats = zip(gearnum, times)
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerows(stats)
        # writer.writerow(time_dict.values())
    print(time_dict)
    # input_parameters(False, True, 6, largest_rear)

def top_gearsets():
    """
    Input:
        Takes result of best_finder() as parquet
        Takes drivetrains as parquet from shifting_pattern()

    Returns:
        Dictionary of best gearset dataframes to plot. Same format as drivetrains
    """
    # Reads in top gearsets parquet
    top_of_top = pd.read_parquet("top_of_top.parquet")

    all_drivetrains = pd.read_parquet("drivetrains_combined.parquet")
    all_drivetrains_df = {name: group.drop(columns="dataset_name") for name, group in all_drivetrains.groupby("dataset_name")}

    # Takes keys from top_of_top and takes only the drivetrains with those keys
    desired_drivetrains = top_of_top.iloc[:,0].tolist()
    filtered_drivetrains = {key: df for key, df in all_drivetrains_df.items() if key in desired_drivetrains}
        
    return(filtered_drivetrains)

def what_to_plot(what_to_plot):
    """
    Performs analysis on real gearsets as spreadsheets did
    """

    # Sets gearsets to whichever wants plotting
    if what_to_plot == "all_analysed":
        all_drivetrains = pd.read_parquet("drivetrains_combined.parquet")
        gearsets = {name: group.drop(columns="dataset_name") for name, group in all_drivetrains.groupby("dataset_name")}
    
    elif what_to_plot == "top_gearsets":
        # Creates instance of store and loads with quadratic params
        gearsets = top_gearsets()

    # Plots ratios across patterned gearset
    graph_shifts(gearsets_to_plot=gearsets)

    return

def main():
    """
    Initialises config file available to all functions, calls functions to run program
    """
    # Starts run timer
    start = time.time()
    print(f"Start time: {start}")
    # Creates instance with config values
    config = GearConfig(max_front=2,
                        max_rear=12,
                        smallest_rear=11,
                        largest_rear=36,
                        smallest_front=40,
                        largest_front=54,
                        use_real=True,
                        use_generated=False)
    # Creates instance of store and loads with quadratic params
    store = cadence_reference()
    
    # Adds peak cadence to store instance
    store = best_cadence(store)
    print(f"Time elapsed: {(time.time()-start)}")
    
    # Checks if score has a cache and runs it if not (can force to run)
    get_data(config,store,"ideal",force_recompute=True)
    print(f"Time elapsed: {(time.time()-start)}")
    
    # # Predicts time taken to run large datasets
    # time_predict()

    # Finds best gears based on measures
    best_finder(config,store,pattern="ideal")
    print(f"Time elapsed: {(time.time()-start)}")

    # Chooses which gearsets to graph ratio shifts on
    what_to_plot("all_analysed")

    # score(config,store,"halves")

    # Plots data using matplotlib
    results_plotter_matplotlib(get_data(config,store,"ideal"))

    # Stops run timer and finds duration
    end = time.time()
    duration = end - start
    print(f"Duration (s): {duration}")

if __name__ == "__main__":
    print("Running main")
    main()