import serial
import cv2
import numpy as np

# --- PnP CONFIGURATION ---
# The Origin (0,0) is now locked perfectly to the CENTER of the quadrilateral.
# +X goes Right. +Y goes Backward (Down the pad towards the bottom LEDs).

object_points = np.array([
    [-28.5, -60.0, 0],  # 0: Top-Left
    [ 47.5, -60.0, 0],  # 1: Top-Right
    [ 49.5,  60.0, 0],  # 2: Bottom-Right
    [-68.5,  60.0, 0]   # 3: Bottom-Left
], dtype=np.float32)

# --- CAMERA INTRINSICS ---
initial_focal_length = 1720
camera_matrix = np.array([
    [initial_focal_length, 0.0, 512.0],
    [0.0, initial_focal_length, 512.0], 
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# --- HELPER FUNCTIONS ---
def candidate_orderings(pts):
    idx = [0, 1, 2, 3]
    orderings = []
    for start in range(4):
        rotated = idx[start:] + idx[:start]
        orderings.append(pts[rotated])
    return orderings

def sort_clockwise(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

def rotation_matrix_to_ypr(R):
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy) 

    if np.abs(np.cos(pitch)) > 1e-6:
        yaw = np.arctan2(-R[0, 2] / np.cos(pitch), R[0, 0] / np.cos(pitch))
        roll = np.arctan2(R[1, 0] / np.cos(pitch), R[1, 1] / np.cos(pitch))
    else:
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# --- POSE SOLVER ---
def solve_pose(image_pts):
    cyclic_pts = sort_clockwise(image_pts.astype(np.float32))
    orderings = candidate_orderings(cyclic_pts)
    results = []

    for start_idx, ordering in enumerate(orderings):
        try:
            retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points, ordering, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
            )
        except cv2.error:
            continue

        if not retval:
            continue

        for rvec, tvec, err in zip(rvecs, tvecs, errors):
            if tvec[2][0] <= 0:
                continue

            err_val = err[0] if hasattr(err, "__len__") else err
            results.append((err_val, rvec, tvec, start_idx))

    if not results:
        return False, None, None, None, None

    results.sort(key=lambda r: r[0])
    best_err, best_rvec, best_tvec, best_start = results[0]
    return True, best_rvec, best_tvec, best_err, orderings[best_start]

cv2.namedWindow("IR Tracking & Pose Estimation")

# --- MAIN LOOP ---
ser = serial.Serial('/dev/ttyACM0', 9600)
print("Visualization running. Press 'q' to quit.")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # 1. Fix the Squish: Use a rectangular aspect ratio (1024 width x 900 height)
            canvas = np.zeros((900, 1024, 3), dtype=np.uint8)
            
            # 2. Always draw the UI background so the window shape is consistent
            cv2.rectangle(canvas, (0, 750), (1024, 900), (30, 30, 30), -1)
            cv2.line(canvas, (0, 750), (1024, 750), (255, 255, 255), 2)

            valid_points = []

            try:
                parts = line.split('|')
                for i, part in enumerate(parts):
                    if '[' in part and ']' in part:
                        coords = part.split('[')[1].split(']')[0].split(',')
                        x, y = int(coords[0]), int(coords[1])
                        
                        if x < 1023 and y < 1023:
                            x = 1023 - x  
                            valid_points.append([x, y])
                            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][i]
                            cv2.circle(canvas, (x, y), 8, color, -1)

                if len(valid_points) >= 3:
                    pts = np.array(valid_points, np.int32)
                    hull = cv2.convexHull(pts)
                    if cv2.contourArea(hull) > 500:  
                        cv2.polylines(canvas, [hull], True, (255, 255, 255), 2)
                    else:
                        valid_points.clear() 

                if len(valid_points) == 4:
                    image_points = np.array(valid_points, dtype=np.float32)

                    success, rvec, tvec, reproj_err, used_ordering = solve_pose(image_points)

                    # Safety check: if reprojection error is massive, reject the pose
                    if success and tvec[2][0] / 10.0 > 2.0 and reproj_err < 15.0:
                        
                        cv2.drawFrameAxes(canvas, camera_matrix, dist_coeffs, rvec, tvec, 50)

                        for point_idx, pt in enumerate(used_ordering):
                            px, py = int(pt[0]), int(pt[1])
                            cv2.putText(canvas, str(point_idx), (px + 15, py - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        rmat, _ = cv2.Rodrigues(rvec)
                        cam_pos_world = -np.dot(rmat.T, tvec)

                        world_x_cm = cam_pos_world[0][0] / 10.0  
                        world_y_cm = cam_pos_world[1][0] / 10.0  
                        world_z_cm = cam_pos_world[2][0] / 10.0  

                        yaw, pitch, roll = rotation_matrix_to_ypr(rmat)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(canvas, f"FOCAL LENGTH: {initial_focal_length}", (380, 780), font, 0.7, (0, 255, 255), 2)

                        pos_text = f"POSITION: X (Red): {world_x_cm:6.1f} cm | Y (Green): {world_y_cm:6.1f} cm | Z (Blue): {world_z_cm:6.1f} cm"
                        rot_text = f"ROTATION: Roll: {roll:6.1f} deg | Pitch: {pitch:6.1f} deg | Yaw: {yaw:6.1f} deg"
                        
                        cv2.putText(canvas, pos_text, (50, 830), font, 0.6, (0, 255, 0), 2)
                        cv2.putText(canvas, rot_text, (50, 870), font, 0.6, (200, 100, 255), 2)
                    else:
                        cv2.putText(canvas, "POSE UNRELIABLE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(canvas, "TARGET LOST", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 3. Maintain rectangular aspect ratio on resize
                display_canvas = cv2.resize(canvas, (1024, 900))
                cv2.imshow("IR Tracking & Pose Estimation", display_canvas)
                
                if cv2.waitKey(1) == ord('q'):
                    break
            except (IndexError, ValueError):
                continue

except KeyboardInterrupt:
    print("Stopping...")
finally:
    ser.close()
    cv2.destroyAllWindows()