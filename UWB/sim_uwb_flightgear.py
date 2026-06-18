import time
import math
import numpy as np
from pymavlink import mavutil
from scipy.optimize import minimize

class UWBPositionSolver:
    def __init__(self):
        # We shifted the entire room up by 5 meters to account for the KSFO runway!
        self.anchors = {
            "A1": [-5.0, -5.0, 6.0],   # Low anchors (1m above the runway)
            "A2": [ 5.0, -5.0, 15.0],  # High anchors (10m above the runway)
            "A3": [ 5.0,  5.0, 6.0],   # Low anchors (1m above the runway)
            "A4": [-5.0,  5.0, 15.0]   # High anchors (10m above the runway)
        }
        self.last_position = [0.0, 0.0, 5.0] # Drone starts at 5m
        
        # We massively expanded the boundaries so the solver never flatlines!
        self.bounds = [(-20.0, 20.0), (-20.0, 20.0), (0.0, 30.0)]

    def _calculate_total_cost(self, guess_point, measured_distances):
        gx, gy, gz = guess_point
        total_squared_error = 0.0
        for anchor_id, measured_radius in measured_distances.items():
            ax, ay, az = self.anchors[anchor_id]
            theoretical_radius = math.sqrt((gx - ax)**2 + (gy - ay)**2 + (gz - az)**2)
            total_squared_error += (theoretical_radius - measured_radius) ** 2
        return total_squared_error

    def resolve_position(self, uwb_data):
        result = minimize(
            self._calculate_total_cost, self.last_position, 
            args=(uwb_data,), method='L-BFGS-B', bounds=self.bounds
        )
        if result.success:
            self.last_position = result.x
        return self.last_position

def main():
    print("--- Connecting to ArduPilot SITL ---")
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    master.wait_heartbeat()
    print("Heartbeat received! Link active.")

    master.mav.request_data_stream_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 30, 1
    )

    uwb_system = UWBPositionSolver()
    
    # LOWERED NOISE: 1cm of noise for initial testing to guarantee stability
    NOISE_LEVEL = 0.01 

    print("\n[BOOTSTRAP 1] Forcing EKF Global Origin to California...")
    master.mav.set_gps_global_origin_send(
        master.target_system,
        int(37.615 * 1e7),   # Latitude * 1e7
        int(-122.389 * 1e7), # Longitude * 1e7
        0                    # Altitude in millimeters
    )
    time.sleep(1) 

    print("[BOOTSTRAP 2] Sending dummy vision data to lock Z-axis...")
    for _ in range(40): 
        current_time_us = int(round(time.time() * 1000000))
        # Safely sending 0.0 for rotation to prevent simulator crashes
        master.mav.vision_position_estimate_send(
            current_time_us, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0
        )
        time.sleep(0.05)
    
    print("[BOOTSTRAP] Complete! Cleared for flight.\n")

    print("Listening for Ground Truth and feeding Noisy Vision data...")
    
    while True:
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        if not msg:
            continue

        true_x, true_y, true_z = msg.x, msg.y, -msg.z 
        
        noisy_radii = {}
        for anchor_id, anchor_pos in uwb_system.anchors.items():
            exact_distance = math.sqrt(
                (true_x - anchor_pos[0])**2 + 
                (true_y - anchor_pos[1])**2 + 
                (true_z - anchor_pos[2])**2
            )
            noisy_radii[anchor_id] = exact_distance + np.random.normal(0, NOISE_LEVEL)

        resolved_pos = uwb_system.resolve_position(noisy_radii)
        
        # THE FIX: Grab the exact microsecond timestamp from the simulator itself!
        # This perfectly synchronizes your Python "eyes" with the drone's "brain."
        sim_time_us = msg.time_boot_ms * 1000
        
        # Safely sending 0.0 for rotation here as well
        master.mav.vision_position_estimate_send(
            sim_time_us,  # USE THE NEW VARIABLE HERE
            resolved_pos[0], resolved_pos[1], -resolved_pos[2], 
            0.0, 0.0, 0.0 
        )
        
        # Safely sending 0.0 for rotation here as well
        master.mav.vision_position_estimate_send(
            current_time_us, 
            resolved_pos[0], resolved_pos[1], -resolved_pos[2], 
            0.0, 0.0, 0.0 
        )

        print(f"Actual Z: {true_z:.2f}m | Solved Z: {resolved_pos[2]:.2f}m")
        time.sleep(0.05) 

if __name__ == '__main__':
    main()