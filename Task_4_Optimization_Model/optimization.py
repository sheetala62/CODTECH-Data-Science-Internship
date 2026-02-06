from pulp import *

# Create problem
model = LpProblem("Profit_Maximization", LpMaximize)

# Decision variables
A = LpVariable("Product_A", lowBound=0)
B = LpVariable("Product_B", lowBound=0)

# Objective function
model += 40*A + 30*B

# Constraints
model += 2*A + 1*B <= 100
model += 1*A + 1*B <= 80

# Solve
model.solve()

print("Status:", LpStatus[model.status])
print("Produce Product A:", A.varValue)
print("Produce Product B:", B.varValue)
print("Maximum Profit:", value(model.objective))
