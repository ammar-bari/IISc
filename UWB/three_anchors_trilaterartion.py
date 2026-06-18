import serial
import math
from scipy.optimize import minimize

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# ==========================================
# --- HARDWARE CALIBRATION ---
# ==========================================
# This is the physical delay inside the silicon and copper traces.
# Adjust this number during your Anchor 2 calibration test!
ANTENNA_OFFSET_CM = 10.0  

# ==========================================
# 1. THE FILTERS
# ==========================================
class KalmanFilter1D:
    def __init__(self, process_noise, measurement_noise):
        self.q = process_noise
        self.r = measurement_noise
        self.x = None  
        self.p = 1.0   

    def update(self, measurement):
        if self.x is None:
            self.x = measurement 
            return self.x
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

class EMAFilter3D:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.x = None
        self.y = None
        self.z = None

    def update(self, new_x, new_y, new_z):
        if self.x is None:
            self.x, self.y, self.z = new_x, new_y, new_z
        else:
            self.x = (self.alpha * new_x) + ((1 - self.alpha) * self.x)
            self.y = (self.alpha * new_y) + ((1 - self.alpha) * self.y)
            self.z = (self.alpha * new_z) + ((1 - self.alpha) * self.z)
        return self.x, self.y, self.z

# ==========================================
# 2. THE SCIPY TRILATERATION ENGINE
# ==========================================
class UWBPositionSolver:
    def __init__(self):
        # Physical Anchor Coordinates [X, Y, Z] in CENTIMETERS
        self.anchors = {
            "A0": [ 73.0, -180.0, 111.0],
            "A1": [119.0,   26.0,  78.0],
            "A2": [  0.0,    0.0,  78.0] # Your Calibration Anchor
        }
        self.last_position = [0.0, 0.0, 100.0]
        self.bounds = [
            (-1000.0, 1000.0),  
            (-1000.0, 1000.0),  
            (    0.0,  500.0)   
        ]

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
# 3. THE LIVE SERIAL LOOP
# ==========================================
def main():
    print(f"Connecting to UWB Tag on {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, dsrdtr=False, rtscts=False)
        print("Connected! Calibrated 3D Spatial Tracking Active:")
        print("-" * 75)
        
        kf0 = KalmanFilter1D(process_noise=0.05, measurement_noise=2.0)
        kf1 = KalmanFilter1D(process_noise=0.05, measurement_noise=2.0)
        kf2 = KalmanFilter1D(process_noise=0.05, measurement_noise=2.0)
        
        output_smoother = EMAFilter3D(alpha=0.15)
        solver = UWBPositionSolver()
        
        while True:
            raw_bytes = ser.readline()
            
            if raw_bytes and b'CmdM' in raw_bytes:
                byte_array = list(raw_bytes)
                
                if len(byte_array) > 30: 
                    # 1. Extract raw millimeters and convert to centimeters
                    raw_0 = ((byte_array[18] * 256) + byte_array[17]) / 10.0
                    raw_1 = ((byte_array[22] * 256) + byte_array[21]) / 10.0
                    raw_2 = ((byte_array[26] * 256) + byte_array[25]) / 10.0
                    
                    # 2. Apply Hardware Calibration (Subtract Antenna Delay)
                    cal_0 = max(0.0, raw_0 - ANTENNA_OFFSET_CM)
                    cal_1 = max(0.0, raw_1 - ANTENNA_OFFSET_CM)
                    cal_2 = max(0.0, raw_2 - ANTENNA_OFFSET_CM)
                    
                    # 3. Apply 1D Kalman Smoothing
                    r0 = kf0.update(cal_0) if cal_0 > 0 else 0.0
                    r1 = kf1.update(cal_1) if cal_1 > 0 else 0.0
                    r2 = kf2.update(cal_2) if cal_2 > 0 else 0.0
                    
                    if r0 > 0 and r1 > 0 and r2 > 0:
                        live_distances = {"A0": r0, "A1": r1, "A2": r2}
                        
                        # 4. Run Gradient Descent solver
                        raw_x, raw_y, raw_z = solver.resolve_position(live_distances)
                        
                        # 5. Apply Output Smoother
                        smooth_x, smooth_y, smooth_z = output_smoother.update(raw_x, raw_y, raw_z)
                        
                        print(f"\rTag -> X: {smooth_x:7.1f} cm  |  Y: {smooth_y:7.1f} cm  |  Z: {smooth_z:7.1f} cm     ", end="", flush=True)
                
    except Exception as e:
        print(f"\nPort Error: {e}")
    except KeyboardInterrupt:
        print("\nTracking Stopped.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()