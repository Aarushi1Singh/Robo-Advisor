This project implements a Robo-Advisor model that automates portfolio construction using modern portfolio theory (MPT) and quantitative optimization. It analyzes mutual funds/ETFs based on sector, region, and risk, then computes the optimal allocation to maximize returns for a given risk tolerance.

🔹 Features
Fund Selection: Filters and selects funds based on sector, region, and risk profile.
Data Processing: Cleans and structures financial datasets for accurate return and risk calculations.
Risk & Return Metrics: Calculates variance–covariance matrices, expected returns, volatility, and Sharpe ratios.
Optimization: Uses mean-variance optimization and constraint-based optimization (scipy.optimize.minimize, COBYLA) to generate optimal allocations.
Visualization: Plots efficient frontier, risk-return scatterplots, and portfolio allocations.
Reporting: Outputs portfolio weights and performance metrics in a clear, interpretable format.

🛠️ Tech Stack
Python
Libraries: NumPy, Pandas, SciPy, Matplotlib
Excel (data preprocessing)

📈 Methodology
Import and preprocess financial data.
Compute expected returns and risk (variance-covariance).
Apply mean-variance optimization with constraints.
Generate efficient frontier and visualize results.
Recommend final portfolio allocation.
