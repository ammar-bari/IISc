import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# 1. YOUR UWB SOLVER CLASS
# ==========================================
class UWBPositionSolver:
    def __init__(self):
        self.anchors = {
            "A1": [-1.5, -1.0, 0.0],
            "A2": [ 1.5, -1.0, 0.0],
            "A3": [ 0.0,  2.0, 2.0]
        }
        self.last_position = [0.0, 0.0, 1.0]
        self.bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 4.0)]

    def _calculate_total_cost(self, guess_point, measured_distances):
        gx, gy, gz = guess_point
        total_squared_error = 0.0
        
        for anchor_id, measured_radius in measured_distances.items():
            ax, ay, az = self.anchors[anchor_id]
            theoretical_radius = math.sqrt((gx - ax)**2 + (gy - ay)**2 + (gz - az)**2)
            error = theoretical_radius - measured_radius
            total_squared_error += error ** 2
            
        return total_squared_error

    def resolve_position(self, uwb_data):
        result = minimize(
            self._calculate_total_cost, 
            self.last_position, 
            args=(uwb_data,), 
            method='L-BFGS-B', 
            bounds=self.bounds
        )
        if result.success:
            self.last_position = result.x
            return result.x
        else:
            return self.last_position

# ==========================================
# 2. THE MATPLOTLIB VISUALIZATION
# ==========================================
def plot_wireframe_sphere(ax, center, radius, color):
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=0.10) # Made slightly more transparent

def main():
    # 1. Initialize System
    uwb_system = UWBPositionSolver()
    
    # --- NOISE SIMULATION SETUP ---
    # Imagine the drone is physically hovering exactly here:
    true_drone_pos = [0.5, 0.5, 1.5]
    
    # We define how bad the UWB noise is (e.g., 0.3 meters of standard deviation)
    NOISE_LEVEL = 0.30 
    noisy_radii = {}
    
    print("--- Simulating Noisy UWB Data ---")
    for anchor_id, anchor_pos in uwb_system.anchors.items():
        # Calculate the exact perfect distance
        exact_distance = math.sqrt(
            (true_drone_pos[0] - anchor_pos[0])**2 + 
            (true_drone_pos[1] - anchor_pos[1])**2 + 
            (true_drone_pos[2] - anchor_pos[2])**2
        )
        
        # Add random Gaussian noise to simulate hardware inaccuracy
        noise = np.random.normal(0, NOISE_LEVEL)
        noisy_radii[anchor_id] = exact_distance + noise
        
        print(f"{anchor_id} -> Exact: {exact_distance:.2f}m | Noisy Reading: {noisy_radii[anchor_id]:.2f}m")
    
    # ------------------------------

    print("\nCalculating position using noisy data...")
    resolved_pos = uwb_system.resolve_position(noisy_radii)
    
    print(f"True Drone Coordinates   : X:{true_drone_pos[0]:.2f}, Y:{true_drone_pos[1]:.2f}, Z:{true_drone_pos[2]:.2f}")
    print(f"Resolved (Guessed) Coords: X:{resolved_pos[0]:.2f}, Y:{resolved_pos[1]:.2f}, Z:{resolved_pos[2]:.2f}")
    
    # Calculate the total real-world positioning error
    final_error = math.sqrt(sum((t - r)**2 for t, r in zip(true_drone_pos, resolved_pos)))
    print(f"Total Positioning Error  : {final_error:.2f} meters")

    # 2. Setup Matplotlib 3D Figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("UWB Trilateration: Optimization Finding the 'Sweet Spot' in Noisy Data")

    colors = {"A1": 'red', "A2": 'green', "A3": 'blue'}

    # 3. Plot Anchors, Spheres, and Lines
    for anchor_id, anchor_pos in uwb_system.anchors.items():
        radius = noisy_radii[anchor_id]
        color = colors[anchor_id]

        # Anchor locations
        ax.scatter(*anchor_pos, color=color, marker='^', s=150)
        
        # Noisy Spheres
        plot_wireframe_sphere(ax, anchor_pos, radius, color)
        
        # Line from Anchor to the RESOLVED (Guessed) position
        ax.plot(
            [anchor_pos[0], resolved_pos[0]], 
            [anchor_pos[1], resolved_pos[1]], 
            [anchor_pos[2], resolved_pos[2]], 
            color=color, linestyle='--', linewidth=1.5, alpha=0.7
        )

    # 4. Plot the Markers
    # Plot the TRUE position where the drone physically is (Green Dot)
    ax.scatter(*true_drone_pos, color='limegreen', marker='o', s=200, label='True Drone Pos', zorder=5)
    
    # Plot the RESOLVED position from the math engine (Black Star)
    ax.scatter(*resolved_pos, color='black', marker='*', s=350, label='Resolved (Math) Pos', zorder=6)

    # 5. Graph Formatting
    ax.set_xlabel('X Axis (Meters)')
    ax.set_ylabel('Y Axis (Meters)')
    ax.set_zlabel('Z Axis (Meters)')
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, 4])

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
    ax.set_box_aspect([x_range, y_range, z_range]) 

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()