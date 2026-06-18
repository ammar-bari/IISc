import serial

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

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

def main():
    print(f"Connecting to {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, dsrdtr=False, rtscts=False)
        print("Connected! Tri-Anchor Kalman Tracking Active:")
        print("-" * 75)
        
        # Initialize a separate Kalman filter for each anchor
        # Q = 0.001 (slightly faster response), R = 0.05 (handles UWB noise)
        kf0 = KalmanFilter1D(process_noise=0.001, measurement_noise=0.05)
        kf1 = KalmanFilter1D(process_noise=0.001, measurement_noise=0.05)
        kf2 = KalmanFilter1D(process_noise=0.001, measurement_noise=0.05)
        
        while True:
            raw_bytes = ser.readline()
            
            if raw_bytes and b'CmdM' in raw_bytes:
                byte_array = list(raw_bytes)
                
                # Make sure the packet is long enough to contain all three anchors
                if len(byte_array) > 30: 
                    
                    # --- Extract Anchor 0 ---
                    dist_0_mm = (byte_array[18] * 256) + byte_array[17]
                    dist_0_m = dist_0_mm / 1000.0
                    
                    # --- Extract Anchor 1 ---
                    dist_1_mm = (byte_array[22] * 256) + byte_array[21]
                    dist_1_m = dist_1_mm / 1000.0
                    
                    # --- Extract Anchor 2 ---
                    dist_2_mm = (byte_array[26] * 256) + byte_array[25]
                    dist_2_m = dist_2_mm / 1000.0
                    
                    # Apply Kalman Filters (Only if distance is > 0, to ignore unplugged anchors)
                    smooth_0 = kf0.update(dist_0_m) if dist_0_m > 0 else 0.0
                    smooth_1 = kf1.update(dist_1_m) if dist_1_m > 0 else 0.0
                    smooth_2 = kf2.update(dist_2_m) if dist_2_m > 0 else 0.0
                    
                    # Print the live dashboard!
                    print(f"\rA0: {smooth_0:.2f}m  |  A1: {smooth_1:.2f}m  |  A2: {smooth_2:.2f}m      ", end="", flush=True)
                
    except Exception as e:
        print(f"\nPort Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("\n")

if __name__ == "__main__":
    main()