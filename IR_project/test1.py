import serial
import cv2
import numpy as np

# Change this to your Arduino port
ser = serial.Serial('/dev/ttyACM0', 9600) 

# Create the canvas
canvas = np.zeros((768, 1024, 3), dtype=np.uint8)

print("Visualization running. Press 'c' to clear, 'q' to quit.")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            canvas = np.zeros((768, 1024, 3), dtype=np.uint8)
            valid_points = []
            
            try:
                parts = line.split('|')
                for i, part in enumerate(parts):
                    if '[' in part and ']' in part:
                        coords = part.split('[')[1].split(']')[0].split(',')
                        x, y = int(coords[0]), int(coords[1])
                        
                        if x < 1023 and y < 1023:
                            # X-Axis Inversion
                            x = 1023 - x  
                            
                            valid_points.append([x, y])

                            # Draw the colored dots
                            color = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)][i]
                            cv2.circle(canvas, (x, y), 8, color, -1) 
                
                # --- NEW: DRAW THE QUADRILATERAL WITH CONVEX HULL ---
                if len(valid_points) == 4:
                    pts = np.array(valid_points, np.int32)
                    
                    # Wrap a digital "rubber band" around the points to prevent crossing
                    hull = cv2.convexHull(pts)
                    
                    # Draw the hull
                    cv2.polylines(canvas, [hull], isClosed=True, color=(255, 255, 255), thickness=2)
                # ----------------------------------------------------
                
                cv2.imshow("IR Current Detection", canvas)
                
                key = cv2.waitKey(1)
                if key == ord('c'):
                    canvas = np.zeros((768, 1024, 3), dtype=np.uint8)
                if key == ord('q'):
                    break
                    
            except (IndexError, ValueError):
                continue

except KeyboardInterrupt:
    print("Stopping...")
    ser.close()
    cv2.destroyAllWindows()