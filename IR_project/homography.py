import serial
import cv2
import numpy as np

# --- PnP CONFIGURATION ---
# Locked to the exact hardware array. 
# This prevents visual sorting from crossing the wires when you rotate it!
object_points = np.array([
    [-52.75, -53.0,   0], # Index 0 (Red):   Physical Top-Left
    [ 54.25, -53.0,   0], # Index 1 (Green): Physical Top-Right
    [ 51.25,  53.0,   0], # Index 2 (Blue):  Physical Bottom-Right
    [-52.75,  53.0,   0]  # Index 3 (Cyan):  Physical Bottom-Left
], dtype=np.float32)

# For Homography, we must drop the Z coordinate (it calculates plane-to-plane)
object_points_2d = object_points[:, :2]

# Adjusted to a realistic focal length to prevent pitch scaling errors
FOCAL_LENGTH = 400.0
camera_matrix = np.array([
    [FOCAL_LENGTH, 0.0, 512.0],
    [0.0, FOCAL_LENGTH, 384.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.zeros((4,1))

# --- INITIALIZE TRACKING MEMORY ---
prev_rvec = None
prev_tvec = None

# --- SERIAL SETUP ---
ser = serial.Serial('/dev/ttyACM0', 9600)
print("Visualization running with Homography. Press 'q' to quit.")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            # Create the unified single canvas (1024x768)
            canvas = np.zeros((768, 1024, 3), dtype=np.uint8)
            valid_points = []
            
            try:
                parts = line.split('|')
                for i, part in enumerate(parts):
                    if '[' in part and ']' in part:
                        coords = part.split('[')[1].split(']')[0].split(',')
                        x, y = int(coords[0]), int(coords[1])
                        if x < 1023 and y < 1023:
                            
                            # Un-mirror the camera feed 
                            x = 1023 - x 
                            
                            valid_points.append([x, y])

                            # Draw the individual colored LED dots
                            color = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)][i]
                            cv2.circle(canvas, (x, y), 8, color, -1)
                            
                # --- DRAW THE QUADRILATERAL ---
                if len(valid_points) >= 3:
                    pts = np.array(valid_points, np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.polylines(canvas, [hull], True, (255, 255, 255), 2)
                    
                # --- HOMOGRAPHY 6-DoF POSE ESTIMATION ---
                if len(valid_points) == 4:
                    
                    # 1. Feed the RAW hardware sequence. No visual sorting allowed.
                    image_points = np.array(valid_points, dtype=np.float32)
                    
                    # Visual Debugger: Ensures the numbers stay locked to their colors
                    for idx, pt in enumerate(image_points):
                        cv2.putText(canvas, str(idx), (int(pt[0]), int(pt[1]) - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # 2. Calculate the Homography matrix
                    H, _ = cv2.findHomography(object_points_2d, image_points, 0)
                    
                    success = False
                    rvec = None
                    tvec = None
                    
                    if H is not None:
                        # 3. Decompose Homography into possible 3D poses
                        num, rotations, translations, normals = cv2.decomposeHomographyMat(H, camera_matrix)
                        
                        valid_solutions = []
                        
                        # 4. Filter out physically impossible solutions (Behind camera OR glitching into the lens)
                        for idx in range(num):
                            T_sol = translations[idx]
                            R_sol = rotations[idx]
                            N_sol = normals[idx]
                            
                            # THE SANITY CHECK: Depth must be greater than 2.0 cm
                            if T_sol[2][0] > 2.0: 
                                valid_solutions.append((R_sol, T_sol, N_sol))
                        
                        if len(valid_solutions) > 0:
                            # 5. Disambiguate using temporal memory (Stops axes from flipping)
                            if prev_rvec is not None and prev_tvec is not None:
                                prev_rmat, _ = cv2.Rodrigues(prev_rvec)
                                best_sol = min(valid_solutions, key=lambda s: np.linalg.norm(s[1] - prev_tvec) + np.linalg.norm(s[0] - prev_rmat))
                            else:
                                best_sol = max(valid_solutions, key=lambda s: abs(s[2][2][0]))
                                
                            R_best, T_best, _ = best_sol
                            
                            rvec, _ = cv2.Rodrigues(R_best)
                            tvec = T_best
                            success = True
                    
                    if success:
                        prev_rvec = rvec
                        prev_tvec = tvec
                        
                        cv2.drawFrameAxes(canvas, camera_matrix, dist_coeffs, rvec, tvec, 50)
                        
                        # Extract Translation
                        x_cm = tvec[0][0] / 10.0
                        y_cm = tvec[1][0] / 10.0
                        z_cm = tvec[2][0] / 10.0
                        
                        # Extract Rotation
                        rmat, _ = cv2.Rodrigues(rvec) 
                        euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat) 
                        pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]
                        
                        # --- DRAW THE HUD PANEL ---
                        cv2.rectangle(canvas, (0, 648), (1024, 768), (30, 30, 30), -1)
                        cv2.line(canvas, (0, 648), (1024, 648), (255, 255, 255), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(canvas, "6-DoF TARGET TRACKING (HOMOGRAPHY)", (320, 680), font, 0.7, (0, 255, 255), 2)
                        
                        pos_text = f"TRANSLATION: X: {x_cm:6.1f} cm | Y: {y_cm:6.1f} cm | Z: {z_cm:6.1f} cm"
                        rot_text = f"ROTATION: Roll: {roll:6.1f} deg | Pitch: {pitch:6.1f} deg | Yaw: {yaw:6.1f} deg"
                        cv2.putText(canvas, pos_text, (100, 720), font, 0.6, (0, 255, 0), 2)
                        cv2.putText(canvas, rot_text, (100, 750), font, 0.6, (200, 100, 255), 2)
                else:
                    cv2.putText(canvas, "TARGET LOST - Need 4 Points", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    prev_rvec = None
                    prev_tvec = None

                cv2.imshow("IR Tracking & Pose Estimation", canvas)
                if cv2.waitKey(1) == ord('q'):
                    break
            except (IndexError, ValueError):
                continue

except KeyboardInterrupt:
    print("Stopping...")
    ser.close()
    cv2.destroyAllWindows()