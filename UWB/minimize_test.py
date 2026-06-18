from scipy.optimize import minimize

# 1. THE OBJECTIVE FUNCTION (The "Valley" we want to find the bottom of)
def cardboard_cost(x_guess):
    x = x_guess[0]
    
    # Calculate the surface area based on our guess for x
    surface_area = (x**2) + (4000 / x)
    return surface_area
    
# 2. RUN THE OPTIMIZER
result = minimize(
    fun = cardboard_cost,       # The function we want to minimize
    x0 = [5.0],                 # Our initial guess: Let's start with a base of 5cm
    method = 'L-BFGS-B',        # Our walking strategy
    bounds = [(1.0, 50.0)]      # The fence: The base must be between 1cm and 50cm
)

# 3. EXTRACT THE RESULTS
best_x = result.x[0]
best_h = 1000 / (best_x**2)
lowest_cardboard_used = result.fun

print(f"Algorithm Success: {result.success}")
print(f"It took {result.nit} steps to find the answer.")
print(f"Perfect Box Base (x): {best_x:.2f} cm")
print(f"Perfect Box Height (h): {best_h:.2f} cm")
print(f"Minimum Cardboard Used: {lowest_cardboard_used:.2f} cm^2")
