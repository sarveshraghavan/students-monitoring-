import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import smtplib
from datetime import datetime
import csv
import time
from scipy.spatial import distance as dist
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- CONFIG ----------
EAR_THRESHOLD = 0.23
SLEEP_DURATION = 20  # seconds
DISTRACTED_POSE_THRESHOLD = 20
STUDENT_EMAIL = "student1@uni.edu"
PROF_EMAIL = "professor@uni.edu"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your_email@gmail.com"
SMTP_PASS = "your_app_password"  # Use App Password for Gmail

# Status tracking system
STATUS_TRACKING = {
    "ENGAGED": 0,
    "DISTRACTED": 0,
    "SLEEPING": 0,
    "PHONE_USAGE": 0
}
current_status = "UNKNOWN"
last_status_change = time.time()
status_history = []

# ---------- SETUP ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
model = YOLO("yolov10n.pt")

sleep_timer = None
sleeping = False
student_actions = []
status = ""
sleep_timers = {}
sleep_flag = {}
student_id = 1

last_print_time = time.time()
PRINT_INTERVAL = 20  # seconds

# ---------- FUNCTIONS ----------
def eye_aspect_ratio(landmarks):
    eye = [33, 160, 158, 133, 153, 144]
    points = [(landmarks[i].x, landmarks[i].y) for i in eye]
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, frame_shape):
    image_points = np.array([
        (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),
        (landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]),
        (landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]),
        (landmarks[61].x * frame_shape[1], landmarks[61].y * frame_shape[0]),
        (landmarks[291].x * frame_shape[1], landmarks[291].y * frame_shape[0]),
        (landmarks[199].x * frame_shape[1], landmarks[199].y * frame_shape[0])
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (-30.0, 30.0, -30.0),
        (30.0, 30.0, -30.0),
        (0.0, 60.0, -30.0)
    ])

    focal_length = frame_shape[1]
    center = (frame_shape[1] // 2, frame_shape[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    _, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rmat, trans_vec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = angles.flatten()
    return pitch, yaw, roll

def log_incident(status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("incidents.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, student_id, status])

def send_alert(status):
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            subject = f"Alert: Student {student_id} is {status}"
            body = f"Time: {datetime.now()}\nStudent ID: {student_id}\nStatus: {status}"
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(SMTP_USER, PROF_EMAIL, message)
        print(f"📧 Alert sent for S{student_id} — {status}")
    except Exception as e:
        print(f"❌ Email fail: {e}")

def generate_pie_chart():
    labels = [k for k in STATUS_TRACKING.keys()]
    sizes = [v for v in STATUS_TRACKING.values()]
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Student Engagement Distribution")
    plt.savefig("engagement_pie.png")
    plt.close()

def generate_heatmap():
    if not status_history:
        print("⚠️ No status history available for heatmap")
        return
    
    df = pd.DataFrame(status_history, columns=['Timestamp', 'Status'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute
    
    # Create 15-minute time bins
    df['TimeBin'] = df['Hour'].astype(str) + ':' + (df['Minute'] // 15 * 15).astype(str).str.zfill(2)
    
    pivot = pd.crosstab(index=df['TimeBin'], columns=df['Status'], normalize='index')
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot.T, cmap="YlGnBu", annot=True, fmt=".0%")
    plt.title("Engagement Status Over Time")
    plt.ylabel("Engagement Status")
    plt.xlabel("Time of Day (15-min intervals)")
    plt.tight_layout()
    plt.savefig("engagement_heatmap.png")
    plt.close()

# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        annotated_frame = frame.copy()
        results = model(frame, verbose=False)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        color = (255, 255, 255)
        new_status = None

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id].lower()

            # Phone detection
            if label in ["cell phone", "mobile phone", "smartphone"]:
                new_status = "PHONE_USAGE"
                status = f"ID:{student_id} 📱 PHONE DETECTED"
                color = (255, 0, 255)
                log_incident("PHONE USAGE")
                send_alert("PHONE USAGE")
                cv2.putText(annotated_frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                continue

            if label != "person":
                continue

            # Face behavior analysis
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                ear = eye_aspect_ratio(landmarks)
                pitch, _, _ = get_head_pose(landmarks, frame.shape)
                eyes_closed = ear < EAR_THRESHOLD
                facing_screen = abs(pitch) < 15
                head_down = pitch > 25
                chin_sideways = abs(pitch) > DISTRACTED_POSE_THRESHOLD
                face_laid_on_table = (y2 - y1) > frame.shape[0] * 0.4

                if (eyes_closed and facing_screen) or (eyes_closed and head_down and chin_sideways) or (face_laid_on_table and eyes_closed):
                    if student_id not in sleep_timers:
                        sleep_timers[student_id] = current_time
                    else:
                        sleep_duration = current_time - sleep_timers[student_id]
                        if sleep_duration >= SLEEP_DURATION:
                            if not sleep_flag.get(student_id, False):
                                sleep_flag[student_id] = True
                                log_incident("SLEEPING")
                                send_alert("SLEEPING")
                            new_status = "SLEEPING"
                            status = f"ID:{student_id} 😴 SLEEPING"
                            color = (0, 0, 255)
                        else:
                            status = f"ID:{student_id} 👁 Eyes Closed (Tracking)"
                            color = (255, 255, 0)
                else:
                    sleep_timers.pop(student_id, None)
                    sleep_flag[student_id] = False

                    if abs(pitch) > DISTRACTED_POSE_THRESHOLD:
                        new_status = "DISTRACTED"
                        status = f"ID:{student_id} 😵 DISTRACTED"
                        color = (0, 165, 255)
                        log_incident("DISTRACTED")
                    elif ear > EAR_THRESHOLD and facing_screen:
                        new_status = "ENGAGED"
                        status = f"ID:{student_id} ✅ ENGAGED"
                        color = (0, 255, 0)

                cv2.putText(annotated_frame, status, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Update status tracking
        if new_status and new_status != current_status:
            if current_status in STATUS_TRACKING:
                elapsed = current_time - last_status_change
                STATUS_TRACKING[current_status] += elapsed
                status_history.append((last_status_change, current_status))
            current_status = new_status
            last_status_change = current_time
            student_actions.append(current_status)

        # Periodic reporting
        if current_time - last_print_time >= PRINT_INTERVAL:
            if current_status in STATUS_TRACKING:
                elapsed = current_time - last_status_change
                STATUS_TRACKING[current_status] += elapsed
                status_history.append((current_time, current_status))
            
            print("\n=== ⏱ 20s Report ===")
            total_time = sum(STATUS_TRACKING.values()) or 1
            for status, seconds in STATUS_TRACKING.items():
                print(f"{status}: {seconds:.1f}s ({seconds/total_time*100:.1f}%)")
            
            generate_pie_chart()
            last_print_time = current_time

        cv2.imshow("SmartClassroomX 🎥", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Final report
    print("\n=== FINAL REPORT ===")
    if current_status in STATUS_TRACKING:
        elapsed = time.time() - last_status_change
        STATUS_TRACKING[current_status] += elapsed
    
    total_time = sum(STATUS_TRACKING.values()) or 1
    for status, seconds in STATUS_TRACKING.items():
        print(f"{status}: {seconds:.1f}s ({seconds/total_time*100:.1f}%)")
    
    generate_pie_chart()
    generate_heatmap()
    print("📊 Visualizations saved to engagement_pie.png and engagement_heatmap.png")
    
    cap.release()
    cv2.destroyAllWindows()