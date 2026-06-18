import serial
import json
import re

# Configure the serial port
# Since it's the native USB port, it will typically be /dev/ttyACM0
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

def main():
    print(f"Connecting to UWB Anchor on {SERIAL_PORT}...")
    
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Connected! Listening for live distance data...\n")
        
        while True:
            # Read one line of data from the serial buffer
            raw_line = ser.readline()
            
            try:
                # Decode the raw bytes into a string
                line = raw_line.decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue
                
                # Use a regex to extract the JSON portion from the raw string
                # Example raw data: JS006C{"TWR":{"a16":"4096","R":115,"T":0,"D":76,...}}
                json_match = re.search(r'\{.*\}', line)
                
                if json_match:
                    json_string = json_match.group(0)
                    data = json.loads(json_string)
                    
                    # Extract the TWR metrics dictionary
                    twr_data = data.get("TWR", {})
                    
                    # 'D' represents the distance in centimeters
                    distance_cm = twr_data.get("D")
                    tag_id = twr_data.get("T", 0)
                    signal_power = twr_data.get("P", 0)
                    
                    if distance_cm is not None:
                        # Print the cleaned data to the console
                        print(f"Tag [{tag_id}] | Distance: {distance_cm} cm | Signal: {signal_power} dBm")
                        
            except json.JSONDecodeError:
                # Skip lines that are corrupted or incomplete JSON fragments
                continue
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue

    except serial.SerialException:
        print(f"Error: Could not open port {SERIAL_PORT}. Make sure the board is plugged in and no other program (like Arduino IDE) is using it.")
    except KeyboardInterrupt:
        print("\nStopping data stream. Exiting cleanly.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
