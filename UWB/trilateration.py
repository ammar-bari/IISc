import math
from scipy.optimize import minimize

class UWBPositionSolver:
    def __init__(self):
        """
        Initializes the solver with the physical layout of the target hoop.
        """
        # Physical Anchor Coordinates [X, Y, Z] in meters.
        # Anchor 3 is elevated to 2.0m to solve the "Z-Axis Curse" (Vertical Diversity)
        self.anchors = {
            "A1": [-1.5, -1.0, 0.0],
            "A2": [ 1.5, -1.0, 0.0],
            "A3": [ 0.0,  2.0, 2.0]
        }
        
        # We store the last calculated position to use as our "smart guess"
        # for the next calculation. We assume the drone starts at 1m altitude.
        self.last_position = [0.0, 0.0, 1.0]
        
        # Flight volume limits: (min, max) in meters. 
        # Z is locked to > 0.0 so the math never guesses the drone is underground.
        self.bounds = [
            (-5.0, 5.0),  # X-axis limits
            (-5.0, 5.0),  # Y-axis limits
            ( 0.0, 4.0)   # Z-axis limits
        ]

    def _calculate_total_cost(self, guess_point, measured_distances):
        """
        The Master Error Equation.
        This calculates the "Total Cost" (sum of squared errors) for a specific guess.
        """
        gx, gy, gz = guess_point
        total_squared_error = 0.0
        
        # Loop through the raw distances sent by the STM32 Tag
        for anchor_id, measured_radius in measured_distances.items():
            ax, ay, az = self.anchors[anchor_id]
            
            # 1. Calculate Theoretical Radius (Pythagorean Theorem)
            theoretical_radius = math.sqrt((gx - ax)**2 + (gy - ay)**2 + (gz - az)**2)
            
            # 2. Find the Residual (Error)
            error = theoretical_radius - measured_radius
            
            # 3. Square the error and add it to the running total
            total_squared_error += error ** 2
            
        return total_squared_error

    def resolve_position(self, uwb_data):
        """
        Runs the Gradient Descent algorithm to slide our guess down the error valley.
        """
        # Run SciPy's minimizer using our _calculate_total_cost function
        result = minimize(
            self._calculate_total_cost,  # The function to minimize
            self.last_position,          # Our starting guess (last known location)
            args=(uwb_data,),            # The raw data from the UWB chips
            method='L-BFGS-B',           # The specific gradient descent algorithm
            bounds=self.bounds           # The physical room boundaries
        )
        
        if result.success:
            # If the algorithm found the absolute lowest point (gradient = 0)
            self.last_position = result.x
            return result.x
        else:
            # Fallback: if UWB data is hopelessly corrupted, don't crash. 
            # Return the last good coordinate to keep the Pixhawk stable.
            print("[WARNING] Optimization failed. Using last known position.")
            return self.last_position


# ==========================================
# SIMULATION: How to use this class
# ==========================================
if __name__ == "__main__":
    # 1. Initialize our solver class
    uwb_system = UWBPositionSolver()
    
    print("--- 3D Trilateration Engine Started ---")
    
    # 2. Simulate Frame 1 (Data arriving via UART from the drone's Tag)
    incoming_data_t1 = {"A1": 2.14, "A2": 2.14, "A3": 1.73}
    pos_t1 = uwb_system.resolve_position(incoming_data_t1)
    print(f"Time 1 -> X: {pos_t1[0]:.2f}, Y: {pos_t1[1]:.2f}, Z: {pos_t1[2]:.2f}")
    
    # 3. Simulate Frame 2 (Drone moves slightly)
    incoming_data_t2 = {"A1": 2.18, "A2": 2.18, "A3": 1.68}
    pos_t2 = uwb_system.resolve_position(incoming_data_t2)
    print(f"Time 2 -> X: {pos_t2[0]:.2f}, Y: {pos_t2[1]:.2f}, Z: {pos_t2[2]:.2f}")