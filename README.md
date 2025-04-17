# DS5105 Exercise 2 – Causal Inference API

This project implements a causal inference model using the **Rubin Causal Model** and exposes it as a REST API using **Flask**. It is containerized with **Docker** and designed to run in **GitHub Codespaces** or any Docker-compatible environment.

## Question 1
### Estimate

Fits the following linear regression model:

Y_i = α + τ·W_i + β·X_i + ε_i

- `W`: binary treatment (carbon offset program)
- `X`: sustainability spending
- `Y`: observed engagement score

**Returns:**
```json
{
  "intercept_alpha": 95.97,
  "ATE_tau": -9.11,
  "p_value_tau": 0.0004,
  "beta_for_X": 1.51
}
```
Also writes results to output.txt and saves the model as model.pkl.

#### How to use:

1. **Start your container** (if not already running):
   ```bash
   docker run -p 5000:5000 -v $(pwd):/app causal-api

2. **Call the /estimate endpoint** (via curl or browser)
    ```bash
    curl http://localhost:5000/estimate

3. **View returned output** (JSON)

## Question 2
## Predict
This endpoint uses the trained linear regression model to predict stakeholder engagement based on:
- **W**: whether the company participates in carbon offset (1 = yes, 0 = no)
- **X**: the amount of sustainability spending

### Example usage (via GET query)

```bash
curl "http://localhost:5000/predict?W=1&X=20"
```

## Returns
```json
{
  "W": 1.0,
  "X": 20.0,
  "predicted_engagement": 117.16
}
```