import casadi as ca

# Step 1: Create an Opti instance
opti = ca.Opti()

# Step 2: Define decision variables
x = opti.variable(2)  # Two decision variables (x1, x2)

# Step 3: Define a parameter
p = opti.parameter()  # Parameter 'p' to be passed later

# Step 4: Define the objective function
# Minimize (x1 - p)^2 + x2^2
objective = (x[0] - p)**2 + x[1]**2
opti.minimize(objective)

# Step 5: Define the constraint
# Constraint: x1 + x2 = 1
opti.subject_to(x[0] + x[1] == 1)

# Step 6: Create a solver
opti.solver('ipopt')

# Step 7: Extract the solver and wrap it in a CasADi Function
# Define inputs and outputs for the function
nlp_solver_fn = ca.Function('nlp_solver_fn', [p], [opti.debug.value(x, opti.solve())])

# Usage: Solve for a specific value of p by calling the function
p_val = 2.0
sol = nlp_solver_fn(p_val)
print("Optimal solution:", sol)
