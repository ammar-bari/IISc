import time
from pymavlink import mavutil

print("Connecting to SITL via UDP...")
# Connect to ArduPilot SITL
master = mavutil.mavlink_connection('udp:127.0.0.1:14550')

# Wait for heartbeat
master.wait_heartbeat()
print("Connected! Sending static vision estimates at 20Hz...")
print("Press Ctrl+C to stop.")

# Fake coordinate to send (e.g., X=1.0m, Y=2.0m, Z=1.5m)
# Remember: In ArduPilot NED, Z is negative for altitude above ground!
fake_x = 1.0
fake_y = 2.0
fake_z = -1.5 

try:
    while True:
        current_time_us = int(round(time.time() * 1000000))
        
        # Send VISION_POSITION_ESTIMATE (Message ID #328)
        master.mav.vision_position_estimate_send(
            current_time_us,    # Timestamp (microseconds)
            fake_x,             # Global X position (m)
            fake_y,             # Global Y position (m)
            fake_z,             # Global Z position (m)
            0, 0, 0             # Roll, Pitch, Yaw (rad)
        )
        
        # Loop rate: ~20 Hz (ArduPilot expects regular updates)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nTest stopped.")
