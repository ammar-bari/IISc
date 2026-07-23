import time
import socket
from smbus2 import SMBus, i2c_msg

# --- I2C CAMERA CONFIGURATION ---
I2C_BUS = 1         
CAMERA_ADDR = 0x58  
bus = SMBus(I2C_BUS)

# --- NETWORK CONFIGURATION ---
UDP_PORT = 5005
# Set up a UDP socket capable of broadcasting
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def write_2bytes(register, data):
    try:
        msg = i2c_msg.write(CAMERA_ADDR, [register, data])
        bus.i2c_rdwr(msg)
        time.sleep(0.01)
    except OSError as e:
        print(f"Init Write Error on register {hex(register)}: {e}")

def init_camera():
    print("Initializing SEN0158 Camera over I2C...")
    write_2bytes(0x30, 0x01)
    write_2bytes(0x30, 0x08)
    write_2bytes(0x06, 0x90)
    write_2bytes(0x08, 0xC0)
    write_2bytes(0x1A, 0x40)
    write_2bytes(0x33, 0x33)
    time.sleep(0.1)
    print("Camera Ready. Tracking and Broadcasting points...")

def read_raw_points():
    try:
        bus.write_byte(CAMERA_ADDR, 0x36)
        time.sleep(0.002) 
        
        read_msg = i2c_msg.read(CAMERA_ADDR, 16)
        bus.i2c_rdwr(read_msg)
        data = list(read_msg)

        x0 = data[1] + ((data[3] & 0x30) << 4)
        y0 = data[2] + ((data[3] & 0xC0) << 2)
        x1 = data[4] + ((data[6] & 0x30) << 4)
        y1 = data[5] + ((data[6] & 0xC0) << 2)
        x2 = data[7] + ((data[9] & 0x30) << 4)
        y2 = data[8] + ((data[9] & 0xC0) << 2)
        x3 = data[10] + ((data[12] & 0x30) << 4)
        y3 = data[11] + ((data[12] & 0xC0) << 2)

        return x0, y0, x1, y1, x2, y2, x3, y3

    except OSError:
        return 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023

# --- MAIN SETUP ---
init_camera()

# --- MAIN LOOP ---
try:
    while True:
        x0, y0, x1, y1, x2, y2, x3, y3 = read_raw_points()

        # Format the data string exactly as expected by the desktop
        msg = f"P0: [{x0}, {y0}]  |  P1: [{x1}, {y1}]  |  P2: [{x2}, {y2}]  |  P3: [{x3}, {y3}]"
        
        # Print to Pi terminal (optional, but good for debugging)
        print(msg)

        # Broadcast over Wi-Fi! The try/except ensures the Pi loop 
        # NEVER crashes even if the network temporarily drops.
        try:
            sock.sendto(msg.encode('utf-8'), ('255.255.255.255', UDP_PORT))
        except Exception:
            pass

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    bus.close()
    sock.close()