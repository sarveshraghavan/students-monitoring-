def train_and_store_embeddings(dataset_path):
    for img_file in os.listdir(dataset_path):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[⚠] Couldn't read: {img_file}")
            continue

        try:
            faces = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend)
            if len(faces) == 0:
                print(f"[🙅‍♀] No face in: {img_file}")
                continue

            embedding_obj = DeepFace.represent(img_path=img_path, model_name="Facenet", detector_backend=detector_backend)[0]
            embedding = embedding_obj["embedding"]
            name = os.path.splitext(img_file)[0]
            img_b64 = img_to_base64(img)

            students_collection.update_one(
                {"_id": name},
                {"$set": {"embedding": embedding, "image": img_b64}},
                upsert=True
            )
            print(f"[✅] Embedded: {name}")
        except Exception as e:
            print(f"[💀] Error embedding {img_file}: {e}")
