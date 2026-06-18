import serial

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

class KalmanFilter1D:
    def __init__(self, process_noise, measurement_noise):
        """
        process_noise (Q): How fast we expect the object to actually move. 
                           Lower = smoother, but lags on sudden stops/starts.
        measurement_noise (R): How noisy we expect the UWB sensor to be.
                               Higher = trusts the sensor less, relies more on the model.
        """
        self.q = process_noise
        self.r = measurement_noise
        self.x = None  # Estimated distance
        self.p = 1.0   # Estimated error/uncertainty

    def update(self, measurement):
        if self.x is None:
            self.x = measurement # Initialize on first reading
            return self.x

        # 1. Predict Phase
        self.p = self.p + self.q

        # 2. Update Phase (Calculate Kalman Gain)
        k = self.p / (self.p + self.r)
        
        # Calculate new estimate
        self.x = self.x + k * (measurement - self.x)
        
        # Update uncertainty for the next loop
        self.p = (1 - k) * self.p
        
        return self.x

def main():
    print(f"Connecting to {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, dsrdtr=False, rtscts=False)
        print("Connected! Live UWB tracking with 1D Kalman Filter:")
        print("-" * 60)
        
        # --- TUNE YOUR KALMAN FILTER HERE ---
        # Q = 1e-4: We assume the object moves relatively smoothly
        # R = 0.05: We assume the UWB sensor has moderate electrical noise
        kf = KalmanFilter1D(process_noise=0.0001, measurement_noise=0.05)
        
        while True:
            raw_bytes = ser.readline()
            
            if raw_bytes and b'CmdM' in raw_bytes:
                byte_array = list(raw_bytes)
                
                if len(byte_array) > 20: 
                    # Extract raw distance in meters
                    low_byte = byte_array[17]
                    high_byte = byte_array[18]
                    raw_distance_m = ((high_byte * 256) + low_byte) / 1000.0
                    
                    # Pass the raw reading through the Kalman Filter
                    kalman_distance = kf.update(raw_distance_m)
                    
                    # Print both to compare the raw jitter vs the Kalman tracking
                    print(f"\rRaw: {raw_distance_m:.2f}m  -->  Kalman: {kalman_distance:.2f}m       ", end="", flush=True)
                
    except Exception as e:
        print(f"\nPort Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("\n")

if __name__ == "__main__":
    main()