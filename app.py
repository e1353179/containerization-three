from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
app = Flask(__name__)
# Training data
X = np.array([[0], [1], [2], [3], [4]])
y = np.array([0, 2, 4, 6, 8])
model = LinearRegression().fit(X, y)
@app.route("/predict")
def predict():
x = float(request.args.get("x", 0))
y_pred = model.predict([[x]])[0]
# Log prediction
with open("output.txt", "w") as f:
f.write(f"Input x: {x}\nPrediction: {y_pred}\n")
return jsonify({"x": x, "prediction": y_pred})
if __name__ == "__main__":
app.run(host="0.0.0.0", port=5000)
