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

# ---------- SETUP ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Use YOLOv8n instead of v10 if you don't have v10
model = YOLO("yolov8n.pt")  # Changed to more commonly available model

sleep_timers = {}
sleep_flag = {}
student_actions = []
status = ""
student_id = 1

last_print_time = time.time()
PRINT_INTERVAL = 20  # seconds

# ---------- FUNCTIONS ----------
def eye_aspect_ratio(landmarks):
    # Correct indices for MediaPipe's 468-point face mesh
    left_eye = [362, 385, 387, 263, 373, 380]
    points = [(landmarks[i].x, landmarks[i].y) for i in left_eye]
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, frame_shape):
    # Updated indices for MediaPipe's face mesh
    image_points = np.array([
        (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),    # Nose tip
        (landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]),   # Left eye left corner
        (landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]), # Right eye right corner
        (landmarks[61].x * frame_shape[1], landmarks[61].y * frame_shape[0]),    # Left mouth corner
        (landmarks[291].x * frame_shape[1], landmarks[291].y * frame_shape[0]), # Right mouth corner
        (landmarks[199].x * frame_shape[1], landmarks[199].y * frame_shape[0])   # Chin
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (-30.0, -30.0, -30.0),        # Left eye
        (30.0, -30.0, -30.0),         # Right eye
        (-30.0, 30.0, -30.0),        # Left mouth
        (30.0, 30.0, -30.0),          # Right mouth
        (0.0, 60.0, -30.0)            # Chin
    ])

    focal_length = frame_shape[1]
    center = (frame_shape[1] // 2, frame_shape[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return 0, 0, 0
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rmat, trans_vec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)

    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

def log_incident(status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("incidents.csv", "a", newline='') as f:
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
        print(f"\U0001F4E7 Alert sent for S{student_id} — {status}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()
    results = model(frame, verbose=False)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    color = (255, 255, 255)
    status = f"ID:{student_id} 👀 SEARCHING"

    # Phone detection
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = model.names[class_id].lower()

        if label in ["cell phone", "mobile phone", "smartphone"]:
            status = f"ID:{student_id} 📱 PHONE DETECTED"
            color = (255, 0, 255)
            log_incident("PHONE USAGE")
            send_alert("PHONE USAGE")
            student_actions.append("DISTRACTED")
            cv2.putText(annotated_frame, status, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    # Face behavior analysis
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        ear = eye_aspect_ratio(landmarks)
        pitch, yaw, roll = get_head_pose(landmarks, frame.shape)
        current_time = time.time()

        eyes_closed = ear < EAR_THRESHOLD
        head_down = pitch > 25
        distracted_pose = abs(pitch) > DISTRACTED_POSE_THRESHOLD

        if eyes_closed:
            if student_id not in sleep_timers:
                sleep_timers[student_id] = current_time
                status = f"ID:{student_id} 👁 Eyes Closed (Tracking)"
                color = (255, 255, 0)
            else:
                if (current_time - sleep_timers[student_id]) >= SLEEP_DURATION:
                    if not sleep_flag.get(student_id, False):
                        sleep_flag[student_id] = True
                        log_incident("SLEEPING")
                        send_alert("SLEEPING")
                    status = f"ID:{student_id} 😴 SLEEPING"
                    color = (0, 0, 255)
                    student_actions.append("SLEEPING")
                else:
                    status = f"ID:{student_id} 👁 Eyes Closed ({int(SLEEP_DURATION - (current_time - sleep_timers[student_id]))}s)"
                    color = (255, 255, 0)
        else:
            sleep_timers.pop(student_id, None)
            sleep_flag[student_id] = False

            if distracted_pose:
                status = f"ID:{student_id} 😵 DISTRACTED"
                color = (0, 165, 255)
                log_incident("DISTRACTED")
                student_actions.append("DISTRACTED")
            else:
                status = f"ID:{student_id} ✅ ENGAGED"
                color = (0, 255, 0)
                student_actions.append("ENGAGED")

        # Display status at top-left corner instead of on bounding box
        cv2.putText(annotated_frame, status, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Periodic report
    current_time = time.time()
    if current_time - last_print_time >= PRINT_INTERVAL:
        print("\n=== ⏱ 20s Report ===")
        if student_actions:
            filtered = [act for act in student_actions if act != "UNKNOWN"]
            if filtered:
                most_common = Counter(filtered).most_common(1)[0][0]
                print(f"🧑‍🎓 Student {student_id} — Most Common: {most_common}")
            else:
                print("😶 No solid actions recorded.")
        print("=====================\n")
        student_actions.clear()
        last_print_time = current_time

    cv2.imshow("SmartClassroomX 🎥", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()