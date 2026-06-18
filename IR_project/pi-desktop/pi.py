import serial
import cv2
import numpy as np
import socket
import json

# --- NETWORK CONFIGURATION ---
DESKTOP_IP = "192.168.1.100"  # <--- CHANGE THIS TO YOUR DESKTOP'S LOCAL IP
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- PnP CONFIGURATION ---
object_points = np.array([
    [-28.5, -60.0, 0],  
    [ 47.5, -60.0, 0],  
    [ 49.5,  60.0, 0],  
    [-68.5,  60.0, 0]   
], dtype=np.float32)

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
        yaw = np.arctan2(R[0, 2], R[0, 2])
        roll = 0.0
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

# --- SIMPLE EMA SMOOTHING WITH SVD ---
class PoseSmoother:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.rvec = None
        self.tvec = None

    def update(self, rvec, tvec):
        if self.rvec is None or self.tvec is None:
            self.rvec = rvec.copy()
            self.tvec = tvec.copy()
        else:
            self.tvec = self.alpha * tvec + (1.0 - self.alpha) * self.tvec
            R_new, _ = cv2.Rodrigues(rvec)
            R_old, _ = cv2.Rodrigues(self.rvec)
            R_smoothed = self.alpha * R_new + (1.0 - self.alpha) * R_old
            
            U, _, Vt = np.linalg.svd(R_smoothed)
            R_ortho = np.dot(U, Vt)
            self.rvec, _ = cv2.Rodrigues(R_ortho)

        return self.rvec, self.tvec
        
    def reset(self):
        self.rvec = None
        self.tvec = None

# --- ADVANCED POSE SOLVER ---
def solve_pose(image_pts, prev_start_idx=None, prev_tvec=None):
    cyclic_pts = sort_clockwise(image_pts.astype(np.float32))
    orderings = candidate_orderings(cyclic_pts)
    results = []

    for start_idx, ordering in enumerate(orderings):
        try:
            retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points, ordering, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
            )
        except cv2.error:
            continue

        if not retval:
            continue

        for rvec, tvec, err in zip(rvecs, tvecs, errors):
            if tvec[2][0] <= 0:
                continue

            rmat, _ = cv2.Rodrigues(rvec)
            if rmat[2, 2] < 0:
                continue 

            err_val = err[0] if hasattr(err, "__len__") else err
            score = err_val

            if prev_start_idx is not None:
                if start_idx != prev_start_idx:
                    score += 1000.0 

            if prev_tvec is not None:
                dist = np.linalg.norm(tvec - prev_tvec)
                score += (dist * 0.5) 

            results.append((score, err_val, rvec, tvec, start_idx))

    if not results:
        return False, None, None, None, None, None

    results.sort(key=lambda r: r[0])
    best_score, best_err, best_rvec, best_tvec, best_start = results[0]
    
    return True, best_rvec, best_tvec, best_err, best_start, orderings[best_start]


# --- MAIN LOOP ---
ser = serial.Serial('/dev/ttyACM0', 9600)
print(f"Tracking running. Sending telemetry to {DESKTOP_IP}:{UDP_PORT}...")

locked_start_idx = None 
locked_tvec = None
smoother = PoseSmoother(alpha=0.15) 

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
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

                # Hull validation
                if len(valid_points) >= 3:
                    pts = np.array(valid_points, np.int32)
                    hull = cv2.convexHull(pts)
                    if cv2.contourArea(hull) <= 500:  
                        valid_points.clear() 

                payload = {
                    "status": "TARGET LOST",
                    "points": valid_points,
                    "focal_length": initial_focal_length
                }

                if len(valid_points) == 4:
                    image_points = np.array(valid_points, dtype=np.float32)

                    success, rvec, tvec, reproj_err, best_start, used_ordering = solve_pose(
                        image_points, prev_start_idx=locked_start_idx, prev_tvec=locked_tvec
                    )

                    if success and tvec[2][0] / 10.0 > 2.0 and reproj_err < 15.0:
                        locked_start_idx = best_start
                        locked_tvec = tvec
                        s_rvec, s_tvec = smoother.update(rvec, tvec)

                        rmat, _ = cv2.Rodrigues(s_rvec)
                        cam_pos_world = -np.dot(rmat.T, s_tvec)

                        world_x_cm = float(cam_pos_world[0][0] / 10.0)
                        world_y_cm = float(cam_pos_world[1][0] / 10.0)
                        world_z_cm = float(cam_pos_world[2][0] / 10.0)
                        yaw, pitch, roll = rotation_matrix_to_ypr(rmat)

                        # Update payload with tracking data
                        payload.update({
                            "status": "TRACKING",
                            "rvec": s_rvec.tolist(),
                            "tvec": s_tvec.tolist(),
                            "ordering": used_ordering.tolist(),
                            "world_pos": [world_x_cm, world_y_cm, world_z_cm],
                            "ypr": [yaw, pitch, roll]
                        })

                    else:
                        payload["status"] = "POSE UNRELIABLE"
                        locked_start_idx = None 
                        locked_tvec = None
                        smoother.reset() 
                else:
                    locked_start_idx = None 
                    locked_tvec = None
                    smoother.reset() 

                # Blast telemetry to desktop
                sock.sendto(json.dumps(payload).encode('utf-8'), (DESKTOP_IP, UDP_PORT))

            except (IndexError, ValueError) as e:
                continue

except KeyboardInterrupt:
    print("Stopping...")
finally:
    ser.close()
    sock.close()