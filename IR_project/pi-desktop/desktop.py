import cv2
import numpy as np
import socket
import json

# --- NETWORK CONFIGURATION ---
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.5) # So UI doesn't freeze if data stops

# --- CAMERA INTRINSICS (Needed for drawing axes) ---
# Ensure this matches the Pi script
initial_focal_length = 1720
camera_matrix = np.array([
    [initial_focal_length, 0.0, 512.0],
    [0.0, initial_focal_length, 512.0], 
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

cv2.namedWindow("IR Tracking & Pose Estimation")
print(f"Listening for telemetry on UDP port {UDP_PORT}... Press 'q' to quit.")

try:
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            payload = json.loads(data.decode('utf-8'))
        except socket.timeout:
            # If no data is received within timeout, just wait.
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # Create blank UI canvas
        canvas = np.zeros((900, 1024, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 750), (1024, 900), (30, 30, 30), -1)
        cv2.line(canvas, (0, 750), (1024, 750), (255, 255, 255), 2)

        status = payload.get("status", "TARGET LOST")
        points = payload.get("points", [])
        f_len = payload.get("focal_length", initial_focal_length)

        # 1. Draw Raw IR Points & Hull
        if points:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            for i, pt in enumerate(points):
                color = colors[i] if i < len(colors) else (255, 255, 255)
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 8, color, -1)

            if len(points) >= 3:
                pts_array = np.array(points, np.int32)
                hull = cv2.convexHull(pts_array)
                cv2.polylines(canvas, [hull], True, (255, 255, 255), 2)

        # 2. Draw Tracking Data
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"FOCAL LENGTH: {f_len}", (380, 780), font, 0.7, (0, 255, 255), 2)

        if status == "TRACKING":
            rvec = np.array(payload["rvec"], dtype=np.float64)
            tvec = np.array(payload["tvec"], dtype=np.float64)
            ordering = payload["ordering"]
            world_pos = payload["world_pos"]
            ypr = payload["ypr"]

            # Draw Axes
            cv2.drawFrameAxes(canvas, camera_matrix, dist_coeffs, rvec, tvec, 50)

            # Draw Ordering Labels
            for point_idx, pt in enumerate(ordering):
                px, py = int(pt[0]), int(pt[1])
                cv2.putText(canvas, str(point_idx), (px + 15, py - 15), font, 0.8, (255, 255, 255), 2)

            # Draw World Telemetry
            pos_text = f"POSITION: X (Red): {world_pos[0]:6.1f} cm | Y (Green): {world_pos[1]:6.1f} cm | Z (Blue): {world_pos[2]:6.1f} cm"
            rot_text = f"ROTATION: Roll: {ypr[2]:6.1f} deg | Pitch: {ypr[1]:6.1f} deg | Yaw: {ypr[0]:6.1f} deg"
            
            cv2.putText(canvas, pos_text, (50, 830), font, 0.6, (0, 255, 0), 2)
            #cv2.putText(canvas, rot_text, (50, 870), font, 0.6, (200, 100, 255), 2)

        elif status == "POSE UNRELIABLE":
            cv2.putText(canvas, "POSE UNRELIABLE", (30, 40), font, 1, (0, 0, 255), 2)
            
        else:
            cv2.putText(canvas, "TARGET LOST", (30, 40), font, 1, (0, 0, 255), 2)

        # Render Canvas
        display_canvas = cv2.resize(canvas, (1024, 900))
        cv2.imshow("IR Tracking & Pose Estimation", display_canvas)
        
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Shutting down visualization...")
finally:
    sock.close()
    cv2.destroyAllWindows()