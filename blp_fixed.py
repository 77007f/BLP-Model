import pyblp
import numpy as np
import pandas as pd

pyblp.options.digits = 2
pyblp.options.verbose = False

# --- Load built-in example data (cereal market shares, prices, characteristics) ---
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)

# --- Step 1: Plain logit (no random coefficients) as a baseline ---
# NOTE: removed absorb='C(product_ids)' -- combined with sugar/mushy
# (which barely vary within product_ids) this was causing the
# instrument matrix to become collinear / rank-deficient.
logit_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
logit_problem = pyblp.Problem(logit_formulation, product_data)
logit_results = logit_problem.solve()
print(logit_results)

# --- Step 2: Random coefficients (BLP) model ---
# X1: linear characteristics (get simple coefficients) -- no absorb here either
X1_formulation = pyblp.Formulation('0 + prices')
# X2: characteristics with random coefficients (heterogeneous tastes)
X2_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
# Demographics that shift tastes (income, age, child)
agent_formulation = pyblp.Formulation('0 + income + age + child')

problem = pyblp.Problem(
    product_formulations=(X1_formulation, X2_formulation),
    product_data=product_data,
    agent_formulation=agent_formulation,
    agent_data=agent_data
)

print(problem)  # shows dimensions, instruments, etc.

# --- Step 3: Solve via GMM ---
results = problem.solve(
    sigma=np.diag([0.3302, 2.4526, 0.0163, 0.2441]),
    pi=[
        [5.4819, 0, 0.2037, 0],
        [15.8935, -1.2000, 0, 2.6342],
        [-0.2506, 0, 0.0511, 0],
        [1.2650, 0, -0.8091, 0]
    ],
    method='1s',
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5})
)
print(results)

# --- Step 4: Post-estimation ---
elasticities = results.compute_elasticities()
markups = results.compute_markups()
print(elasticities)
print(markups)
