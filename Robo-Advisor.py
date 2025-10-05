import gradio as gr
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------
# Custom CSS for the UI
# -----------------------
custom_css = """
    /* Style for the submit button */
    #submit_btn {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    #submit_btn:hover {
        background-color: #3e8e41 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Style for vertical radio options with uniform block height */
    .gradio-radio .option {
        display: block;
        width: 100%;
        padding: 10px 15px;
        margin-bottom: 5px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        transition: all 0.2s ease;
    }
    .gradio-radio .option:hover {
        background-color: #e6e6e6;
        border-color: #aaa;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Basic styling for headings and overall font */
    h1, h2, h3, body {
        font-family: 'Arial', sans-serif;
    }
    
    /* App header styling */
    .app-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #4CAF50, #2196F3);
        margin-bottom: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Landing page styling */
    .features-grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
        margin: 30px 0;
    }
    
    .feature-card {
        flex: 1;
        min-width: 250px;
        max-width: 350px;
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .feature-card h3 {
        color: #4CAF50;
        margin-top: 0;
        margin-bottom: 10px;
    }
    
    .feature-card p {
        color: #333;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }
    
    .steps-container {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        max-width: 800px;
        margin: 20px auto;
        border-left: 5px solid #4CAF50;
    }
    
    /* Styling for the risk profile header */
    .risk-header {
        text-align: center;
        padding: 15px;
        background-color: #eef;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    
    /* Summary Cards Styling */
    .summary-container {
        font-family: 'Arial', sans-serif;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .summary-section {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 25px;
    }
    
    .summary-title {
        font-size: 22px;
        color: #333;
        margin-top: 0;
        margin-bottom: 15px;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }
    
    .cards-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .stat-card {
        flex: 1;
        min-width: 180px;
        background: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stat-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    
    .stat-value {
        font-size: 22px;
        font-weight: bold;
        color: #333;
    }
    
    .description-card {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border-left: 4px solid #4CAF50;
    }
    
    .utility-section {
        background: #f8f8ff;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        font-family: monospace;
        line-height: 1.6;
    }
    
    .confidence-interval {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(to right, #ff9966, #66ccff);
        height: 40px;
        border-radius: 20px;
        padding: 5px;
        margin: 20px 0;
        position: relative;
    }
    
    .confidence-marker {
        position: absolute;
        top: -25px;
        transform: translateX(-50%);
        text-align: center;
    }
    
    .confidence-value {
        font-weight: bold;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        background: rgba(0,0,0,0.1);
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .weights-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .weight-item {
        flex: 1;
        min-width: 180px;
        background: #f9f9f9;
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
    }
    
    .weight-value {
        font-weight: bold;
        color: #4CAF50;
    }
    
    .highlight-best {
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .highlight-worst {
        color: #C62828;
        background-color: #FFEBEE;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* For small screens */
    @media (max-width: 768px) {
        .stat-card {
            min-width: 140px;
        }
        
        .weight-item {
            min-width: 140px;
        }
    }
"""

# Add these styles to your custom_css
chart_cards_css = """
    /* Chart Cards Styling */
    .chart-cards-container {
        display: flex;
        flex-direction: column;
        gap: 25px;
        margin: 20px 0;
    }
    
    .chart-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .chart-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transform: translateY(-5px);
    }
    
    .chart-title {
        font-size: 18px;
        color: #333;
        margin-top: 0;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }
    
    .chart-description {
        font-size: 14px;
        color: #666;
        margin-bottom: 15px;
        line-height: 1.5;
    }
    
    @media (max-width: 768px) {
        .chart-cards-container {
            flex-direction: column;
        }
    }
"""

# Update your custom_css variable to include the chart cards styling
custom_css = custom_css + chart_cards_css
# ======= Portfolio-optimization code ===================

# varcov_table = np.array([
#     [0.000146468, 6.51709E-05, 6.67167E-05, 2.62092E-05, 4.99049E-05, 7.10934E-05, 0.000110946, 8.23428E-05, 5.20107E-05, 7.91214E-05, -1.29962E-06, 0.000110403],
#     [6.51709E-05, 0.000103329, 6.26506E-05, 2.77612E-05, 6.55392E-05, 7.16446E-05, 0.000100388, 8.615E-05, 6.55492E-05, 6.51519E-05, -3.66581E-07, 0.000191111],
#     [6.67167E-05, 6.26506E-05, 0.000100267, 2.76011E-05, 4.11785E-05, 6.39122E-05, 8.60997E-05, 0.000105589, 4.92681E-05, 6.03153E-05, -5.46791E-07, 0.000172119],
#     [2.62092E-05, 2.77612E-05, 2.76011E-05, 0.000118636, 2.50606E-05, 1.83174E-05, 2.68625E-05, 3.45512E-05, 1.6628E-05, 2.98324E-05, -1.10867E-06, 0.000109223],
#     [4.99049E-05, 6.55392E-05, 4.11785E-05, 2.50606E-05, 0.000132924, 5.62156E-05, 0.00011904, 6.6844E-05, 8.26318E-05, 4.33239E-05, -4.95412E-07, 9.92802E-05],
#     [7.10934E-05, 7.16446E-05, 6.39122E-05, 1.83174E-05, 5.62156E-05, 0.000201447, 0.000121437, 5.82713E-05, 8.99742E-05, 6.80401E-05, -9.28095E-07, 0.000161402],
#     [0.000110946, 0.000100388, 8.60997E-05, 2.68625E-05, 0.00011904, 0.000121437, 0.000575939, 0.000185579, 0.000114688, 9.75077E-05, -8.5721E-07, 0.000347231],
#     [8.23428E-05, 8.615E-05, 0.000105589, 3.45512E-05, 6.6844E-05, 5.82713E-05, 0.000185579, 0.000710831, 6.37663E-05, 7.67082E-05, 4.15547E-06, 0.000140885],
#     [5.20107E-05, 6.55492E-05, 4.92681E-05, 1.6628E-05, 8.26318E-05, 8.99742E-05, 0.000114688, 6.37663E-05, 0.000164183, 5.12426E-05, -1.0268E-06, 0.000169502],
#     [7.91214E-05, 6.51519E-05, 6.03153E-05, 2.98324E-05, 4.33239E-05, 6.80401E-05, 9.75077E-05, 7.67082E-05, 5.12426E-05, 0.000141765, 1.70117E-07, 0.000165273],
#     [-1.29962E-06, -3.66581E-07, -5.46791E-07, -1.10867E-06, -4.95412E-07, -9.28095E-07, -8.5721E-07, 4.15547E-06, -1.0268E-06, 1.70117E-07, 2.26502E-07, 2.26926E-07],
#     [0.000110403, 0.000191111, 0.000172119, 0.000109223, 9.92802E-05, 0.000161402, 0.000347231, 0.000140885, 0.000169502, 0.000165273, 2.26926E-07, 0.00020496],
# ])

varcov_table=np.array([
[0.00005783252358,      0.0000008188491971,     0.00002599956474,       0.000005448083118,      -0.000001516929059,     0.000004363999129,      0.000002184729088,      -0.00000007075999813,   -0.00000188478169,      0.000003182968299],
[0.0000008188491971,    0.00005742707494,       0.0000003757472749,     0.00001553740121,       0.0000008756910494,     0.000005044106279,      0.000007903862419,      -0.00000004082585577,   0.00001114824033,       0.00002433468151],
[0.00002599956474,      0.0000003757472749,     0.00009010471285,       -0.000001859353838,     -0.000002694923473,     0.000001479002793,      0.0000002796358603,     -0.000001214069714,     -0.000004600932283,     0.000002684461178],
[0.000005448083118,     0.00001553740121,       -0.000001859353838,     0.0002199085024,        0.000003852611596,      0.00008661382437,       0.000009666604951,      0.000002555821641,      0.00008924999677,       0.00002635948809],
[-0.000001516929059,    0.0000008756910494,     -0.000002694923473,     0.000003852611596,      0.00004339089246,       0.000002358908064,      0.0000006022483796,     0.0000005346512427,     0.000000762723043,      0.000002845257992],
[0.000004363999129,     0.000005044106279,      0.000001479002793,      0.00008661382437,       0.000002358908064,      0.0000640536051,        0.000006313738098,      0.000001651418512,      0.00002882278815,       0.00001264273636],
[0.000002184729088,     0.000007903862419,      0.0000002796358603,     0.000009666604951,      0.0000006022483796,     0.000006313738098,      0.00004600806636,       -0.000001487486851,     0.000004275884941,      0.00001146731816],
[-0.00000007075999813,  -0.00000004082585577,   -0.000001214069714,     0.000002555821641,      0.0000005346512427,     0.000001651418512,      -0.000001487486851,     0.00006801925769,       -0.0000005304047828,    -0.000001833126628],
[-0.00000188478169,     0.00001114824033,       -0.000004600932283,     0.00008924999677,       0.000000762723043,      0.00002882278815,       0.000004275884941,      -0.0000005304047828,    0.0003082520662,        0.0000128923708],
[0.000003182968299,     0.00002433468151,       0.000002684461178,      0.00002635948809,       0.000002845257992,      0.00001264273636,       0.00001146731816,       -0.000001833126628,     0.0000128923708,        0.00009545711433],])

varcov_table = varcov_table * 252  # Annualize the covariance matrix

fund_name=["AB SICAV I American Growth A SGD",
           "United ASEAN Fund SGD",
           "AB SICAV I International Health Care Portfolio A SGD",
           "Blackrock Next Generation technology A2 SGD-H",
           "FTIF - Templeton Global Income A Mdis SGD-H1",
           "Eastspring Investments - Global Dynamic Growth Equity A SGD-H",
           "Natixis Mirova Global Sustainable Equity H-R/A-NPF (SGD)",
           "Stewart Investors Worldwide Leaders A Acc SGD",
           "BlackRock World Energy Fund A2 SGD-H",
           "Eastspring Investment Unit Trusts – Pan European SGD",]

avg_ret=np.array([0.00009235282996,
                  0.00005267556367,
                  0.000328381945,
                  0.000311216066,
                  -0.0000936635957,
                  0.0001073630148,
                  0.000210555295,
                  0.0003144258907,
                  0.0001890057797,
                  0.0002949047289]
)

avg_ret = avg_ret * 252  # Annualize the returns


# avg_ret = np.array([
#     -2.31223E-05,
#     0.000332029,
#     -6.6703E-05,
#     0.000381164,
#     0.000204621,
#     0.00084679,
#     9.47326E-06,
#     0.001860331,
#     0.000750128,
#     -0.000176028,
#     0.000249085,
#     0.001317962
# ])


# -----------------------
# Portfolio Calculation Functions
# -----------------------
# def find_return(x, risk_aversion):
#     last_weight = 1 - np.sum(x)
#     weight_arr = np.append(x, last_weight)
#     # Use all returns except the last element if needed
#     # return np.matmul(weight_arr, avg_ret[:-1])
#     return np.matmul(weight_arr, avg_ret)
def find_return(x, avg_ret):
    last_weight = 1 - np.sum(x)
    weight_arr = np.append(x, last_weight)
    return np.dot(weight_arr, avg_ret)


# def find_standard_dev(x, risk_aversion):
#     last_weight = 1 - np.sum(x)
#     weight_arr = np.append(x, last_weight)
#     # Compute portfolio variance then return it (later we take sqrt for standard dev)
#     return np.matmul(weight_arr, np.matmul(varcov_table[:-1, :-1], weight_arr))

def find_standard_dev(x, varcov_table):
    last_weight = 1 - np.sum(x)
    weight_arr = np.append(x, last_weight)
    return np.matmul(weight_arr, np.matmul(varcov_table, weight_arr))  # Return variance (no sqrt)


def find_variance(x, varcov_table):
    return find_standard_dev(x, varcov_table)

def objective_function(x, risk_aversion, avg_ret, varcov_table):
    return (risk_aversion / 2) * find_variance(x, varcov_table) - find_return(x, avg_ret)


def constraint1(x):
    return 1 - np.sum(x)

def optimize_portfolio(risk_aversion):
    initial_weight = np.array([0.001]*(len(fund_name)-1))
    cons = [{'type': 'ineq', 'fun': constraint1}]
    bnds = tuple((0, 1) for _ in range(len(fund_name)-1))
    
    result = minimize(
        fun=lambda x: objective_function(x, risk_aversion, avg_ret, varcov_table),
        x0=initial_weight,
        method='COBYLA',
        bounds=bnds,
        constraints=cons
    )
    
    x_opt = result.x
    last_weight = 1 - np.sum(x_opt)
    weights = np.append(x_opt, last_weight)
    
    portfolio_return = find_return(x_opt, avg_ret)
    portfolio_variance = find_variance(x_opt, varcov_table)
    portfolio_std = np.sqrt(portfolio_variance)
    obj_value = objective_function(x_opt, risk_aversion, avg_ret, varcov_table)
    portfolio_utility = -1 * obj_value
    
    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Risk Aversion Parameter (A): {risk_aversion:.6f}")
    print(f"Portfolio Return (r): {portfolio_return:.8f}")
    print(f"Portfolio Variance (σ²): {portfolio_variance:.8f}")
    print(f"Portfolio Std Dev (σ): {portfolio_std:.8f}")
    print(f"Objective Function Value: {obj_value:.8f}")
    print(f"Utility = r - (A/2)σ² = {portfolio_utility:.8f}")
    print("Optimal Weights:")
    for name, weight in zip(fund_name, weights):
        print(f"  {name}: {weight:.6f}")
    print("------------------------\n")
    
    best_return = portfolio_return + 2 * portfolio_std
    worst_return = portfolio_return - 2 * portfolio_std
    median_return = portfolio_return
    
    data = {
        "weights": weights,
        "return": portfolio_return,
        "std": portfolio_std,
        "variance": portfolio_variance,
        "utility": portfolio_utility,
        "objective_value": obj_value,
        "best_return": best_return,
        "worst_return": worst_return,
        "median_return": median_return,
        "risk_aversion": risk_aversion
    }
    return data

# -----------------------
# Efficient Frontier Calculation - find initial frontier
# -----------------------

# Efficient frontier calculations (without short sale constraint)
def objective_function_eff_front(x, varcov_table):
    return find_standard_dev(x, varcov_table)**0.5 


# def constraint2(x, avg_ret, needed_return, small_number=1E-12):
#     return small_number - abs(needed_return - find_return(x, avg_ret))

def constraint2(x, avg_ret, needed_return,small_number=1E-12):
    return small_number-abs(needed_return-find_return(x,avg_ret))


def compute_efficient_frontier_data():
    """Compute efficient frontier data - called lazily when needed"""
    initial_weight_min_2=np.array([0.001]*(len(fund_name)-1))

    up_bound=max(avg_ret)
    low_bound=min(avg_ret)
    observation=25
    interval=(up_bound-low_bound)/observation

    eff_frontier_ret=[low_bound + (i*interval) for i in range(0,observation+1)]

    eff_frontier_weight=[]
    eff_frontier_standard_dev=[]
    curr_initial_weight=initial_weight_min_2
    for needed_return in eff_frontier_ret:
        cons =[{'type':'ineq', 'fun': constraint1},
               {'type':'ineq', 'fun': lambda x: constraint2(x, avg_ret, needed_return)}]
        bnd=  tuple(((0, 1) for _ in range((len(fund_name)-1))))
        result = minimize(fun=lambda x: objective_function_eff_front(x, varcov_table), x0=curr_initial_weight, method='COBYLA', bounds=bnd, constraints=cons)
        eff_frontier_weight.append(result.x)
        standard_dev=find_variance(result.x, varcov_table)**0.5
        eff_frontier_standard_dev.append(standard_dev)
        curr_initial_weight=result.x

    # Efficient frontier calculation with short selling allowed (without bounds)
    eff_frontier_weight_w_ss=[]
    eff_frontier_standard_dev_w_ss=[]
    curr_initial_weight=initial_weight_min_2
    for needed_return in eff_frontier_ret:
        cons =[{'type':'ineq', 'fun': lambda x: constraint2(x, avg_ret, needed_return)}]
        result = minimize(fun=lambda x: objective_function_eff_front(x, varcov_table), x0=curr_initial_weight, method='COBYLA', constraints=cons)
        eff_frontier_weight_w_ss.append(result.x)
        standard_dev=find_variance(result.x, varcov_table)**0.5
        eff_frontier_standard_dev_w_ss.append(standard_dev)
        curr_initial_weight=result.x
    
    return eff_frontier_ret, eff_frontier_standard_dev, eff_frontier_standard_dev_w_ss

# Initialize as None - will be computed lazily
eff_frontier_ret = None
eff_frontier_standard_dev = None
eff_frontier_standard_dev_w_ss = None

# Compute fund risks from variance (take only the first len(fund_name) diagonal elements)
fund_risks = np.sqrt(np.diag(varcov_table)[:len(fund_name)])

# Create a list of dictionaries for individual funds
individual_ret_data = []
for i, name in enumerate(fund_name):
    individual_ret_data.append({'name': name, 'risk': fund_risks[i], 'return': avg_ret[i]})

# Convert the list into a pandas DataFrame
df_funds = pd.DataFrame(individual_ret_data)
df_funds.set_index('name', inplace=True)

# -----------------------
# Efficient Frontier Calculation for Gradio Purpose
# -----------------------
    
def compute_efficient_frontier():
    # Sweep a range of risk-aversion parameters to plot the efficient frontier.
    A_values = np.linspace(0.8, 10, 20)
    frontier_returns = []
    frontier_stds = []
    for A in A_values:
        res = optimize_portfolio(A)
        frontier_returns.append(res["return"])
        frontier_stds.append(np.sqrt(res["variance"]))
    return A_values, frontier_stds, frontier_returns

def plot_efficient_frontier():
    A_values, frontier_stds, frontier_returns = compute_efficient_frontier()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_stds, frontier_returns, marker='o', label="Efficient Frontier")
    
    # Plot all user profile risk/return points as red dots.
    if all_profiles:
        profiles_std = [p["Std"] for p in all_profiles]
        profiles_return = [p["Return"] for p in all_profiles]
        ax.scatter(profiles_std, profiles_return, color="red", s=100, zorder=5, label="Profile Return")
    
    ax.set_xlabel("Portfolio Risk (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier with Profile Returns")
    ax.legend()
    return fig

def plot_eff_new():
    # Compute efficient frontier data lazily
    global eff_frontier_ret, eff_frontier_standard_dev, eff_frontier_standard_dev_w_ss
    if eff_frontier_ret is None:
        eff_frontier_ret, eff_frontier_standard_dev, eff_frontier_standard_dev_w_ss = compute_efficient_frontier_data()
    
    # Create a figure and axes using subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the efficient frontier 
    ax.plot(eff_frontier_standard_dev_w_ss, eff_frontier_ret, marker="o", label="With Short Sale")
    ax.plot(eff_frontier_standard_dev, eff_frontier_ret, marker="X", label="Without Short Sale")

    # Plot individual funds using the DataFrame
    ax.scatter(df_funds['risk'], df_funds['return'], marker="s", color="red", label="Individual Fund")
 
    # Plot profile points if they exist
    if not profile_df.empty:
        ax.scatter(profile_df["risk"], profile_df["return"], color="green", label="Profile Return")

        # Annotate each profile point
        for fund, row in profile_df.iterrows():
            ax.annotate(fund, (row['risk'], row['return']), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Annotate each fund's point with its name using the DataFrame's index (fund name)
    for fund, row in df_funds.iterrows():
        ax.annotate(fund, (row['risk'], row['return']), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Standard Deviation (Risk)')
    ax.set_ylabel('Return')
    ax.set_title('Efficient Frontier with Individual Fund Returns')
    ax.legend()

    # Remove plt.show() as it's causing issues with Gradio
    # plt.show()  # Comment out or remove this line

    # Return the figure object for further usage or saving
    return fig

# -----------------------
# Risk Assessment System
# -----------------------
risk_questions = [
    {
        "question": "1. What is your primary financial goal for this investment?",
        "options": [
            {"text": "Protect principal; I cannot tolerate losses.", "points": 1},
            {"text": "Keep up with inflation while protecting most of my principal.", "points": 2},
            {"text": "Grow my capital moderately over time.", "points": 3},
            {"text": "Achieve high growth, accepting significant risk.", "points": 4},
            {"text": "Maximize returns aggressively, even at the risk of large losses.", "points": 5}
        ]
    },
    {
        "question": "2. How long do you plan to keep this money invested before you'll need a significant portion of it?",
        "options": [
            {"text": "Less than 3 years", "points": 1},
            {"text": "3-5 years", "points": 2},
            {"text": "5-10 years", "points": 3},
            {"text": "More than 10 years", "points": 4}
        ]
    },
    {
        "question": "3. How familiar are you with investing in stocks, bonds, or mutual funds/ETFs?",
        "options": [
            {"text": "Very little knowledge; I'm a novice.", "points": 1},
            {"text": "Some basic knowledge but limited hands-on experience.", "points": 2},
            {"text": "Moderate; I have invested before and understand how markets move.", "points": 3},
            {"text": "Advanced; I actively follow markets and have used a range of strategies.", "points": 4}
        ]
    },
    {
        "question": "4. Have you experienced a significant market downturn while invested?",
        "options": [
            {"text": "No, and I'm worried about how I'd react.", "points": 1},
            {"text": "Yes, but I sold quickly to avoid large losses.", "points": 2},
            {"text": "Yes, and I held or adjusted my positions calmly.", "points": 3},
            {"text": "Yes, and I actually took advantage of the downturn to buy more.", "points": 4}
        ]
    },
    {
        "question": "5. Imagine your portfolio falls by 15% over a few months. How would you react?",
        "options": [
            {"text": "I would sell immediately to avoid further losses.", "points": 1},
            {"text": "I would reduce some exposure but stay mostly invested.", "points": 2},
            {"text": "I would hold my positions until the market recovers.", "points": 3},
            {"text": "I would buy more, seeing this as an opportunity.", "points": 4}
        ]
    },
    {
        "question": "6. Which statement best reflects your attitude toward short-term market volatility?",
        "options": [
            {"text": "I want minimal fluctuation, even if it means minimal returns.", "points": 1},
            {"text": "Some ups and downs are acceptable, but I don't want large swings.", "points": 2},
            {"text": "Volatility is normal—I can handle moderate drawdowns for higher returns.", "points": 3},
            {"text": "I embrace significant volatility if it may lead to higher long-term gains.", "points": 4}
        ]
    },
    {
        "question": "7. Which best describes your income stability?",
        "options": [
            {"text": "Very stable (e.g., secure salary or pension).", "points": 1},
            {"text": "Moderately stable (some variability in income).", "points": 2},
            {"text": "Unstable or unpredictable.", "points": 3}
        ]
    },
    {
        "question": "8. Do you rely on investment returns to cover near-term expenses (in the next 3-5 years)?",
        "options": [
            {"text": "Yes, I need the income soon.", "points": 1},
            {"text": "Not much, but I can't afford big losses.", "points": 2},
            {"text": "No, I can keep money invested for the long run.", "points": 3}
        ]
    },
    {
        "question": "9. How old are you, and how far are you from retirement (or the date you need this capital)?",
        "options": [
            {"text": "Under 35 (15+ years away from major withdrawals).", "points": 1},
            {"text": "35-50 (still 10+ years to invest).", "points": 2},
            {"text": "50-60 (retirement or major expenses in under 10 years).", "points": 3},
            {"text": "60+ (retiree or close to it).", "points": 4}
        ]
    },
    {
        "question": "10. Which scenario do you find most acceptable for a 1-year period?",
        "options": [
            {"text": "A 5% chance of losing more than 5% of your portfolio, potential annual return up to ~4-5%.", "points": 1},
            {"text": "A 5% chance of losing more than 15%, potential annual return up to ~6-8%.", "points": 2},
            {"text": "A 5% chance of losing more than 25%, potential annual return up to ~10-12%.", "points": 3},
            {"text": "A 5% chance of losing more than 35%, potential annual return up to ~15-20%.", "points": 4}
        ]
    }
]

risk_levels = [
    {"range": (0, 7), "level": "Low (Level 1)", "group": "Very defensive", "description": "Minimal risk; prioritizes capital preservation.", "risk_aversion": 10.0},
    {"range": (8, 15), "level": "Low (Level 2)", "group": "Defensive", "description": "Willing to accept a little more fluctuation but still quite conservative.", "risk_aversion": 8.0},
    {"range": (16, 19), "level": "Medium (Level 3)", "group": "Conservative", "description": "Moderate risk tolerance, balanced approach.", "risk_aversion": 6.0},
    {"range": (20, 25), "level": "Medium (Level 3)", "group": "Moderate", "description": "Moderate risk tolerance, balanced approach.", "risk_aversion": 4.0},
    {"range": (26, 30), "level": "Medium (Level 4)", "group": "Moderately aggressive", "description": "High risk tolerance, expecting higher returns with bigger ups and downs.", "risk_aversion": 2.5},
    {"range": (31, 35), "level": "Medium (Level 4)", "group": "Aggressive", "description": "High risk tolerance, expecting higher returns with bigger ups and downs.", "risk_aversion": 1.5},
    {"range": (36, 100), "level": "High (Level 5)", "group": "Very aggressive", "description": "Significant or very high risk; comfortable with substantial volatility and potential losses.", "risk_aversion": 0.8}
]

def calculate_risk_aversion(responses):
    total_points = 0
    for resp in responses:
        try:
            if isinstance(resp, int):
                total_points += (resp + 1)
            elif isinstance(resp, str) and resp.isdigit():
                total_points += int(resp)
        except (ValueError, TypeError):
            continue
    
    print(f"Calculated total points: {total_points} from responses: {responses}")
    
    matching_level = None
    for level in risk_levels:
        if level["range"][0] <= total_points <= level["range"][1]:
            matching_level = level
            break
    if matching_level is None:
        print(f"No matching risk level found for points: {total_points}, defaulting to Moderate")
        matching_level = risk_levels[3]
    else:
        print(f"Matched to risk level: {matching_level['group']} with points: {total_points}")
    
    return {
        "total_points": total_points,
        "risk_level": matching_level["level"],
        "risk_group": matching_level["group"],
        "description": matching_level["description"],
        "risk_aversion": matching_level["risk_aversion"]
    }

all_profiles = []
profile_df = pd.DataFrame(columns=["name", "risk", "return"])

risk_icons = {
    "Very defensive": "🛡️",
    "Defensive": "🔒",
    "Conservative": "🤔",
    "Moderate": "🙂",
    "Moderately aggressive": "⚡️",
    "Aggressive": "🚀",
    "Very aggressive": "🔥"
}

# -----------------------
# Submit Profile & Return Figures
# -----------------------
def submit_profile(name, *quiz_responses):
    if not name.strip():
        return "<div class='risk-header' style='background-color:#ffebee;'><h2>⚠️ Please enter your name</h2></div>", None, None, None, None
        
    risk_assessment = calculate_risk_aversion(quiz_responses)
    risk_aversion = risk_assessment["risk_aversion"]
    result = optimize_portfolio(risk_aversion)
    
    portfolio_return = result["return"]
    portfolio_variance = result["variance"]
    utility_value = result["utility"]
    manual_utility = portfolio_return - (risk_aversion / 2) * portfolio_variance
    
    # Save profile; note we also store "Std" (risk) and "Return" for plotting on the frontier.
    profile = {
        "Name": name,
        "QuizAnswers": quiz_responses,
        "RiskAssessment": risk_assessment,
        "RiskAversion": risk_aversion,
        "Weights": result["weights"],
        "Return": result["return"],
        "Std": result["std"],
        "Variance": result["variance"],
        "Utility": result["utility"],
        "Best": result["best_return"],
        "Worst": result["worst_return"],
        "Median": result["median_return"]
    }
    
    profile_for_eff = {
        "name": name,
        "risk": result["std"],
        "return": result["return"],
    }
    all_profiles.append(profile)
    
    # Update profile_df as before
    global profile_df
    new_row = pd.DataFrame([profile_for_eff]).set_index('name')
    profile_df = pd.concat([profile_df, new_row])
    
    # Get risk icon
    risk_icon = risk_icons.get(risk_assessment["risk_group"], "")
    
    # Calculate annualized metrics
    annual_return = result['return']  # Already annualized 
    annual_std = result['std']  # Already annualized
    annual_best = result['best_return']  # Already annualized
    annual_worst = result['worst_return']  # Already annualized
    
    # Create beautified HTML summary
    summary_html = f"""
    <div class="summary-container">
        <h1 style="text-align: center; margin-bottom: 30px;">Portfolio Optimization for {name} 🎯</h1>
        
        <!-- Risk Assessment Section -->
        <div class="summary-section">
            <h2 class="summary-title">Risk Assessment Results</h2>
            
            <div class="cards-container">
                <div class="stat-card">
                    <div class="stat-label">Total Score</div>
                    <div class="stat-value">{risk_assessment['total_points']} points</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Risk Level</div>
                    <div class="stat-value">{risk_assessment['risk_level']}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Risk Group</div>
                    <div class="stat-value">{risk_icon} {risk_assessment['risk_group']}</div>
                </div>
            </div>
            
            <div class="description-card">
                <strong>Description:</strong> {risk_assessment['description']}
            </div>
            
            <div class="stat-card" style="margin-top: 15px;">
                <div class="stat-label">Risk Aversion Parameter (A)</div>
                <div class="stat-value">{risk_aversion:.2f}</div>
            </div>
        </div>
        
        <!-- Portfolio Results Section -->
        <div class="summary-section">
            <h2 class="summary-title">Portfolio Results</h2>
            
            <div class="cards-container">
                <div class="stat-card">
                    <div class="stat-label">Expected Annual Return</div>
                    <div class="stat-value" style="color: #4CAF50;">{annual_return:.2%}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Annual Std Dev (σ)</div>
                    <div class="stat-value">{annual_std:.2%}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Annual Variance (σ²)</div>
                    <div class="stat-value">{annual_std**2:.4f}</div>
                </div>
            </div>
            
            <div class="utility-section">
                <strong>Utility Function:</strong><br>
                U(r,σ) = {portfolio_return:.4f} - ({risk_aversion/2:.4f} × {portfolio_variance:.4f})<br>
                = <strong>{manual_utility:.4f}</strong><br>
                Optimizer Utility: <strong>{utility_value:.4f}</strong>
            </div>
            
            <h3 style="margin-top: 25px;">95% Confidence Interval (Annual Returns)</h3>
            <div class="confidence-interval">
                <div class="confidence-marker" style="left: 10%;">
                    <div class="confidence-value highlight-worst">{annual_worst:.2%}</div>
                    <div>Worst</div>
                </div>
                
                <div class="confidence-marker" style="left: 50%;">
                    <div class="confidence-value">{annual_return:.2%}</div>
                    <div>Mean</div>
                </div>
                
                <div class="confidence-marker" style="left: 90%;">
                    <div class="confidence-value highlight-best">{annual_best:.2%}</div>
                    <div>Best</div>
                </div>
            </div>
        </div>
        
        <!-- Portfolio Weights Section -->
        <div class="summary-section">
            <h2 class="summary-title">Optimal Portfolio Weights</h2>
            
            <div class="weights-container">
    """
    
    # Add weight items dynamically
    for f, w in zip(fund_name, result["weights"]):
        # Skip very small allocations
        if w < 0.005:
            continue
            
        # Add color intensity based on weight
        bg_intensity = min(0.9, 0.1 + w * 0.8)
        bg_color = f"rgba(76, 175, 80, {bg_intensity})"
        text_color = "white" if w > 0.2 else "black"
        
        summary_html += f"""
        <div class="weight-item" style="background-color: {bg_color}; color: {text_color};">
            <div>{f}</div>
            <div class="weight-value">{w:.2%}</div>
        </div>
        """
    
    # Close the weights container and summary section
    summary_html += """
            </div>
        </div>
    """
    
    # Generate the charts as before
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    sizes = []
    for f, w in zip(fund_name, result["weights"]):
        if w >= 0.0001:
            labels.append(f)
            sizes.append(w)
    
    # Use a colorful palette with explode effect for largest slice
    colors = plt.cm.tab20(np.linspace(0, 1, len(sizes)))
    explode = np.zeros(len(sizes))
    if len(sizes) > 0:
        explode[np.argmax(sizes)] = 0.1
    
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, 
           colors=colors, explode=explode, shadow=True)
    ax.set_title(f"Optimal Portfolio Allocation ({risk_icon} {risk_assessment['risk_group']})")
    ax.axis("equal")
    
    mean = result['median_return']
    std = result['std']
    worst = result['worst_return']
    best = result['best_return']
    
    x = np.linspace(mean - 4*std, mean + 4*std, 500)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(x, y, 'b-', lw=2, label="Normal Distribution")
    ax2.fill_between(x, y, where=(x >= worst) & (x <= best), color="skyblue", alpha=0.5, label="95% Confidence Interval")
    ax2.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")
    ax2.axvline(worst, color="orange", linestyle="--", label=f"Worst: {worst:.4f}")
    ax2.axvline(best, color="green", linestyle="--", label=f"Best: {best:.4f}")
    ax2.set_title("Portfolio Return Distribution with 95% Confidence Interval")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Probability Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x_values = [best, worst, mean]
    t_values = np.arange(0, 21)
    y_values = [10000 * (1 + x_val) ** t_values for x_val in x_values]
    upper_bound = np.max(y_values, axis=0)
    lower_bound = np.min(y_values, axis=0)
    for i, x_val in enumerate(x_values):
        label = ["Best", "Worst", "Mean"][i]
        ax3.plot(t_values, y_values[i], color=["green", "red", "blue"][i], lw=2, label=f"{label}: {x_val:.4f}")
        ax3.text(t_values[-1], y_values[i][-1], f"${y_values[i][-1]:,.0f}", 
                fontsize=10, va='bottom', color=["green", "red", "blue"][i])
    ax3.fill_between(t_values, lower_bound, upper_bound, color='gray', alpha=0.2, label="Range")
    ax3.set_xlabel("Years")
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.set_title("Projected Portfolio Value: $10,000 × (1 + x)^t")
    ax3.legend()
    ax3.grid(True)
    
    # Format y-axis as currency
    from matplotlib.ticker import FuncFormatter
    def currency_formatter(x, pos):
        return f"${x:,.0f}"
    ax3.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Remove plt.show() as it's causing issues with Gradio
    # plt.show()  # Comment out or remove this line
    
    # Compute the updated efficient frontier plot
    frontier_fig = plot_eff_new()
    
    return summary_html, fig, fig2, fig3, frontier_fig

def refresh_profiles():
    if len(all_profiles) == 0:
        return pd.DataFrame({"No profiles yet": []})
    records = []
    for p in all_profiles:
        risk_icon = risk_icons.get(p["RiskAssessment"]["risk_group"], "")
        row = {
            "Name": p["Name"],
            "Risk Group": f"{risk_icon} {p['RiskAssessment']['risk_group']}",
            "Risk Score": p["RiskAssessment"]["total_points"],
            "Risk Aversion (A)": p["RiskAversion"],
            "Return (r)": p["Return"],
            "Std Dev (σ)": p["Std"],
            "Variance (σ²)": p.get("Variance", p["Std"]**2),
            "Utility U(r,σ)": p["Utility"],
            "Best": p["Best"],
            "Worst": p["Worst"]
        }
        records.append(row)
    df = pd.DataFrame(records)
    return df


# -----------------------
# GRADIO UI with Icons, Landing Page and Beautiful Summaries
# -----------------------
with gr.Blocks(css=custom_css) as demo:
    
    # Landing page/header at the top
    gr.HTML("""
    <div class="app-header">
        <h1 style="margin: 0;">🤖 Enhanced Robo Investor App</h1>
        <p style="margin-top: 10px;">Get personalized investment recommendations based on your risk profile</p>
    </div>
    
    <div style="text-align: center; margin-bottom: 30px;">
        <div class="features-grid">
            <div class="feature-card">
                <h3>📊 Personalized Portfolio</h3>
                <p>Take our risk assessment and get a customized investment portfolio 
                that matches your risk tolerance and financial goals.</p>
            </div>
            
            <div class="feature-card">
                <h3>📈 Advanced Analytics</h3>
                <p>Explore the efficient frontier and see how your portfolio compares 
                to optimal asset allocations based on risk and return.</p>
            </div>
            
            <div class="feature-card">
                <h3>👥 Profile Management</h3>
                <p>Create and compare multiple investment profiles for different 
                financial goals or family members.</p>
            </div>
        </div>
        
        <div class="steps-container">
            <h3>How it works:</h3>
            <ol style="text-align: left; padding-left: 20px;">
                <li>Take our quick risk assessment questionnaire</li>
                <li>Our algorithm analyzes your risk tolerance and goals</li>
                <li>Receive a personalized investment portfolio</li>
                <li>Explore your potential returns and portfolio analytics</li>
            </ol>
        </div>
        
        <div style="margin: 20px 0;">
            <h3>Risk Profiles:</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 10px;">
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">🛡️ Very defensive</div>
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">🔒 Defensive</div>
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">🤔 Conservative</div>
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">🙂 Moderate</div>
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">⚡️ Moderately aggressive</div>
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">🚀 Aggressive</div>
                <div style="background: #f8f8ff; padding: 8px 15px; border-radius: 20px;">🔥 Very aggressive</div>
            </div>
        </div>
    </div>
    """)
    
    # Tabs with icons
    with gr.Tab("📊 Take Risk Assessment"):
        gr.Markdown("## Enter your details to get a customized portfolio recommendation")
        
        with gr.Row():
            name_input = gr.Textbox(label="Your Name")
        
        quiz_inputs = []
        for q in risk_questions:
            with gr.Group():
                gr.Markdown(f"**{q['question']}**")
                radio = gr.Radio(
                    choices=[opt["text"] for opt in q["options"]],
                    label="",
                    value=None,
                    type="index"  # index returned; add 1 later for actual points
                )
                quiz_inputs.append(radio)
        
        with gr.Row():
            submit_btn = gr.Button("Submit Assessment", elem_id="submit_btn")
        
        # First show the summary HTML
        output_area = gr.HTML(label="Portfolio Summary")
        
        # Then show each chart separately with a proper label
        gr.Markdown("### Portfolio Charts")
        
        with gr.Row():
            with gr.Column():
                output_chart = gr.Plot(label="Portfolio Allocation")
            with gr.Column():
                output_bell = gr.Plot(label="Return Distribution")
                
        with gr.Row():
            with gr.Column():
                output_return = gr.Plot(label="Projected Portfolio Growth")
            with gr.Column():
                output_frontier = gr.Plot(label="Efficient Frontier")
        
        submit_btn.click(
            fn=submit_profile,
            inputs=[name_input] + quiz_inputs,
            outputs=[output_area, output_chart, output_bell, output_return, output_frontier]
        )
    
    with gr.Tab("📈 Efficient Frontier"):
        gr.Markdown("## Efficient Frontier")
        
        with gr.Row():
            frontier_btn = gr.Button("Plot Efficient Frontier", elem_id="submit_btn")
        
        with gr.Column():
            frontier_plot = gr.Plot(label="Efficient Frontier Visualization")
        
        frontier_btn.click(
            fn=plot_eff_new,
            inputs=[],
            outputs=frontier_plot
        )
        
    with gr.Tab("👤 All Profiles"):
        gr.Markdown("## List of all User Profiles")
        
        with gr.Row():
            refresh_btn = gr.Button("Refresh List", elem_id="submit_btn")
        
        with gr.Column():
            df_out = gr.Dataframe(
                headers=["Name", "Risk Group", "Risk Score", "Risk Aversion (A)", "Return (r)", "Std Dev (σ)", "Variance (σ²)", "Utility U(r,σ)", "Best", "Worst"],
                label="Profiles",
                value=pd.DataFrame()
            )
        
        refresh_btn.click(
            fn=refresh_profiles,
            inputs=[],
            outputs=df_out
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)
