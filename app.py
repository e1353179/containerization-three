from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib

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

@app.route("/")
def index():
    return "Flask app is running. Visit /estimate to calculate ATE or /predict to estimate Ŷ."

@app.route("/estimate")
def estimate_ate():
    try:
        # Add intercept term
        X_mat = sm.add_constant(df[['W', 'X']])
        y = df['Y']

        # Fit regression model
        model = sm.OLS(y, X_mat).fit()

        # Extract parameters
        alpha = model.params['const']
        tau = model.params['W']
        beta = model.params['X']
        p_value_tau = model.pvalues['W']

        # Save to file
        with open("output.txt", "w") as f:
            f.write(f"Intercept (α): {alpha:.2f}\n")
            f.write(f"ATE (τ): {tau:.2f}, p-value: {p_value_tau:.4f}\n")
            f.write(f"Beta (X): {beta:.2f}\n")

        # Save model to disk
        joblib.dump(model, "model.pkl")

        return jsonify({
            "intercept_alpha": round(alpha, 2),
            "ATE_tau": round(tau, 2),
            "p_value_tau": round(p_value_tau, 4),
            "beta_for_X": round(beta, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET"])
def predict_engagement():
    try:
        # 获取 URL 查询参数，如 /predict?W=1&X=20
        W = float(request.args.get("W") or request.args.get("w") or 0)
        X_val = float(request.args.get("X") or request.args.get("x") or 0)

        # 加载模型
        model = joblib.load("model.pkl")

        # 添加截距项
        input_features = np.array([[1, W, X_val]])
        prediction = model.predict(input_features)[0]

        return jsonify({
            "W": W,
            "X": X_val,
            "predicted_engagement": round(prediction, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
