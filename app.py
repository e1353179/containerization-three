from flask import Flask, jsonify
import numpy as np
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)

# === Simulated data from the exercise ===
data = {
    'Y': [137,118,124,124,120,129,122,142,128,114,
          132,130,130,112,132,117,134,132,121,128],
    'W': [0,1,1,1,0,1,1,0,0,1,
          1,0,0,1,0,1,0,0,1,1],
    'X': [19.8,23.4,27.7,24.6,21.5,25.1,22.4,29.3,20.8,20.2,
          27.3,24.5,22.9,18.4,24.2,21.0,25.9,23.2,21.6,22.8]
}
df = pd.DataFrame(data)

@app.route("/estimate")
def estimate_ate():
    # Add intercept column to feature matrix
    X_mat = sm.add_constant(df[['W', 'X']])
    y = df['Y']

    # Fit the linear regression model
    model = sm.OLS(y, X_mat).fit()

    # Extract estimated parameters
    alpha = model.params['const']     # Intercept
    tau = model.params['W']           # ATE
    beta = model.params['X']          # Spending effect
    p_value_tau = model.pvalues['W']  # Significance of ATE

    # Save output to file
    with open("output.txt", "w") as f:
        f.write(f"Intercept (α): {alpha:.2f}\n")
        f.write(f"ATE (τ): {tau:.2f}, p-value: {p_value_tau:.4f}\n")
        f.write(f"Beta (X): {beta:.2f}\n")

    return jsonify({
        "intercept_alpha": round(alpha, 2),
        "ATE_tau": round(tau, 2),
        "p_value_tau": round(p_value_tau, 4),
        "beta_for_X": round(beta, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
