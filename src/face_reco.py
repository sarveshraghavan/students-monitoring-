import os
import cv2
import numpy as np
import base64
from datetime import datetime
from deepface import DeepFace
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# 💾 MongoDB Setup2
client = MongoClient("mongodb://localhost:27017/")
db = client['smartclassroom']
students_collection = db['students']
attendance_collection = db['attendance']

# 🧠 Detector Backend
detector_backend = "retinaface"

# 🛠 Utils
def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_img(base64_str):
    img_data = base64.b64decode(base64_str)
    return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

# 💾 Save Embeddings to Mongo
def train_and_store_embeddings(dataset_path, detector_backend="opencv"):
    for img_file in os.listdir(dataset_path):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[⚠] Couldn't read: {img_file}")
            continue

        try:
            # 🧠 Detect and Represent Face
            faces = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend)
            if len(faces) == 0:
                print(f"[🙅‍♀️] No face detected in: {img_file}")
                continue

            embedding_obj = DeepFace.represent(img_path=img_path, model_name="Facenet", detector_backend=detector_backend)[0]
            embedding = embedding_obj["embedding"]

            # ✨ Prepare Data
            name = os.path.splitext(img_file)[0]
            img_b64 = img_to_base64(img)
            embedding = list(map(float, embedding))  # ensure it's JSON serializable

            # 💾 Save to MongoDB
            result = students_collection.update_one(
                {"_id": name},
                {"$set": {
                    "embedding": embedding,
                    "image": img_b64
                }},
                upsert=True
            )

            print(f"[✅] {name} saved to DB | Matched: {result.matched_count}, Upserted ID: {result.upserted_id}")

        except Exception as e:
            print(f"[💀] Error embedding {img_file}: {e}")
# 🧠 Compare Embeddings
def compare_embeddings(e1, e2):
    e1 = np.array(e1).reshape(1, -1)
    e2 = np.array(e2).reshape(1, -1)
    return float(np.dot(e1, e2.T) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

# 🕵 Recognize from Frame
def recognize_face(img):
    try:
        temp_img_path = "temp_frame.jpg"
        cv2.imwrite(temp_img_path, img)

        faces = DeepFace.extract_faces(img_path=temp_img_path, detector_backend=detector_backend, enforce_detection=False)
        if len(faces) == 0:
            print("🥲 No face detected.")
            return

        face_img = faces[0]['face']
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, face_img)

        embedding_obj = DeepFace.represent(img_path=temp_face_path, model_name="Facenet", detector_backend=detector_backend)[0]
        current_embedding = embedding_obj["embedding"]

        best_match = None
        highest_similarity = 0.5

        for student in students_collection.find():
            known_embedding = student['embedding']
            similarity = compare_embeddings(current_embedding, known_embedding)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = student

        if best_match:
            print(f"[🎯] Recognized: {best_match['_id']} ({highest_similarity*100:.2f}%)")
            mark_attendance(best_match['_id'])
            matched_img = base64_to_img(best_match['image'])
            cv2.imshow("Matched Face", matched_img)
            cv2.putText(img, best_match['_id'], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            print("😶 Unknown face detected")

    except Exception as e:
        print(f"[💥] Error in recognition: {e}")

# 📆 Attendance Logger
def mark_attendance(name):
    now = datetime.now()
    attendance_collection.insert_one({
        "name": name,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d")
    })
    print(f"[📝] Attendance marked for: {name}")

# 🎥 Live Attendance
def live_attendance():
    cap = cv2.VideoCapture(0)
    print("[📹] Live camera started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        recognize_face(frame)
        cv2.imshow("Live Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_face_embedding(image_path, detector_backend="opencv"):
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend=detector_backend,
            enforce_detection=True
        )[0]
        return np.array(embedding_obj["embedding"])
    except Exception as e:
        print(f"[❌] Embedding failed: {e}")
        return None

# --- Face Matching Logic ---
def compare_with_database(image_path, threshold=0.7):
    target_embedding = get_face_embedding(image_path)
    if target_embedding is None:
        print("[🚫] Could not process input image.")
        return

    max_score = -1
    best_match_name = None

    print("\n🔍 Matching with database faces...\n")

    for doc in students_collection.find():
        name = doc["_id"]
        db_embedding = np.array(doc["embedding"])
        score = cosine_similarity([target_embedding], [db_embedding])[0][0]

        print(f"🧠 Compared with: {name} → Similarity: {round(score * 100, 2)}%")

        if score > max_score:
            max_score = score
            best_match_name = name

    print("\n🧾 Final Verdict:")
    if max_score >= threshold:
        print(f"✅ It's a match! This is likely")
        print(f"🎯 Match Accuracy: {round(max_score * 100, 2)}% 💯")
    else:
        print("❌ No familiar face found in database 😶‍🌫️")


# 🚀 Let's Go!
if __name__ == "__main__":
    dataset_path = r"C:\Users\Nandhini Prakash\smartclassroom\opencv\attendance\dataset\karthi"
    #train_and_store_embeddings(dataset_path)
    image_to_check = "dataset\karthi\chk1.jpg"  # replace with the image path
    compare_with_database(image_to_check)
    #live_attendance()