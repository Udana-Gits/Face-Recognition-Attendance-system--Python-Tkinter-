#code of the MODEL

import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
import tensorflow as tf
import threading
import time
from mtcnn import MTCNN
import sys


def initialize_system():
    """Initialize the attendance system by creating necessary files."""
    # Create attendance file if it doesn't exist
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Index No", "Name"])
        df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
        print(f"✅ Created attendance file: {ATTENDANCE_FILE}")

    # Ensure dataset directory exists
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Load embeddings
    load_student_embeddings()



# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Paths
DATASET_PATH = "known_faces/"
ATTENDANCE_FILE = "attendance.xlsx"

# Ensure dataset directory exists
os.makedirs(DATASET_PATH, exist_ok=True)

# ✅ Global variable for detected faces
detected_faces = []


# Initialize Face Detector
detector = MTCNN()



# ✅ Register a new student by capturing 5 images
def register_student(name, index_number, video_capture):
    if not name or not index_number:
        return "Error: Name and Index Number required."

    student_folder = os.path.join(DATASET_PATH, f"{index_number}_{name}")
    os.makedirs(student_folder, exist_ok=True)

    if not video_capture.isOpened():
        print("⚠ Camera was closed. Reopening...")
        video_capture.open(0)

    # Define poses to capture with instructions
    poses = [
        {"name": "front", "count": 0, "target": 10, "instruction": "Look forward at the camera"},
        {"name": "right", "count": 0, "target": 10, "instruction": "Turn your head to the RIGHT"},
        {"name": "left", "count": 0, "target": 10, "instruction": "Turn your head to the LEFT"}
    ]

    current_pose_index = 0
    current_pose = poses[current_pose_index]
    countdown_start = None
    countdown_value = 3  # Initial countdown time in seconds
    embeddings = []
    is_countdown = True  # Start with countdown mode
    capture_interval = 0.3  # Time between captures in seconds
    last_capture_time = 0

    # Process continues until all poses are captured
    while current_pose_index < len(poses):
        current_pose = poses[current_pose_index]

        ret, frame = video_capture.read()
        if not ret:
            print("⚠ Error: Could not read frame from camera. Retrying...")
            continue

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face using MTCNN
        faces = detector.detect_faces(rgb_frame)

        # Draw instruction and progress
        instruction_text = current_pose["instruction"]
        progress_text = f"Progress: {current_pose['count']}/{current_pose['target']}"

        # Add instructions to the frame
        cv2.putText(frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Process face if detected
        if faces:
            face = faces[0]
            x, y, w, h = face['box']
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check face orientation
            landmarks = face['keypoints']
            is_correct_pose = check_face_orientation(landmarks, current_pose["name"])

            # Draw orientation status
            orientation_text = "Correct Pose" if is_correct_pose else "Adjust Position"
            color = (0, 255, 0) if is_correct_pose else (0, 0, 255)
            cv2.putText(frame, orientation_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Handle countdown and capturing
            current_time = time.time()

            if is_countdown:
                # Countdown mode
                if is_correct_pose:
                    if countdown_start is None:
                        countdown_start = current_time

                    elapsed = current_time - countdown_start
                    remaining = max(0, countdown_value - int(elapsed))

                    # Show countdown
                    cv2.putText(frame, f"Starting in: {remaining}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # When countdown finishes
                    if remaining == 0:
                        is_countdown = False
                        last_capture_time = current_time
                else:
                    # Reset countdown if pose changes
                    countdown_start = None
            else:
                # Capture mode - take photos at regular intervals
                if is_correct_pose and (current_time - last_capture_time >= capture_interval):
                    # Extract and save face
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    cropped_face = rgb_frame[y:y + h, x:x + w]

                    if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                        # Resize for consistency
                        cropped_face = cv2.resize(cropped_face, (112, 112))

                        # Save image
                        img_path = os.path.join(student_folder,
                                                f"{current_pose['name']}_{current_pose['count'] + 1}.jpg")
                        cv2.imwrite(img_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

                        # Extract embedding
                        try:
                            embedding = \
                            DeepFace.represent(cropped_face, model_name="ArcFace", enforce_detection=False)[0][
                                "embedding"]
                            embeddings.append(embedding)
                        except Exception as e:
                            print(f"⚠ Could not extract embedding: {e}")
                            continue

                        current_pose["count"] += 1
                        last_capture_time = current_time

                        # Show capture indicator
                        cv2.putText(frame, "Capturing", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                        print(
                            f"✅ Captured {current_pose['name']} pose: {current_pose['count']}/{current_pose['target']}")

                # Check if we've completed the current pose
                if current_pose["count"] >= current_pose["target"]:
                    current_pose_index += 1
                    # Reset to countdown mode for next pose
                    is_countdown = True
                    countdown_start = None
                    if current_pose_index < len(poses):
                        print(f"Moving to next pose: {poses[current_pose_index]['name']}")
        else:
            cv2.putText(frame, "No face detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Reset countdown if face lost
            if is_countdown:
                countdown_start = None

        # Display the processed frame (for debugging/development)
        cv2.imshow("Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save mean embedding
    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
        np.save(os.path.join(student_folder, "embedding.npy"), mean_embedding)
        print(f"✅ Stored embeddings for {name} ({index_number})")

    return f"✅ Registration Completed: {name} ({index_number})"


# Helper function to check face orientation based on landmarks
def check_face_orientation(landmarks, required_pose):
    """
    Determine if the face orientation matches the required pose
    based on facial landmarks from MTCNN
    """
    # Get key points
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    nose = landmarks['nose']

    # Calculate horizontal distance between nose and center of eyes
    eyes_center_x = (left_eye[0] + right_eye[0]) / 2
    eyes_center_y = (left_eye[1] + right_eye[1]) / 2

    # Calculate eye line angle (orientation)
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Calculate horizontal nose offset
    nose_offset = nose[0] - eyes_center_x

    # Calculate the distance between eyes (for scale normalization)
    eye_distance = np.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)

    # Normalize the nose offset by eye distance
    normalized_offset = nose_offset / eye_distance

    # Determine pose based on nose position relative to eyes center
    if required_pose == "front":
        # For front pose: nose should be aligned with center of eyes
        # and eye line should be relatively horizontal
        return abs(normalized_offset) < 0.15 and abs(eye_angle) < 10

    elif required_pose == "right":
        # For right pose: nose should be to the left of eyes center
        # (from camera's perspective, it's the person's right)
        return normalized_offset < -0.2

    elif required_pose == "left":
        # For left pose: nose should be to the right of eyes center
        return normalized_offset > 0.2

    return False



# ✅ Detect faces asynchronously
def detect_faces_async(frame):
    """Threaded function to detect faces without blocking UI."""
    global detected_faces
    detected_faces = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)



# ✅ Load embeddings once and use them for comparison
student_embeddings = {}


# In face_recognition_model.py
def load_student_embeddings():
    """Optimized version with better error handling."""
    global student_embeddings
    student_embeddings.clear()

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
        return

    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Index No", "Name"])
        df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")

    # Load all student embeddings
    student_count = 0
    for student_folder in os.listdir(DATASET_PATH):
        student_path = os.path.join(DATASET_PATH, student_folder)

        if not os.path.isdir(student_path):
            continue

        embedding_path = os.path.join(student_path, "embedding.npy")

        if os.path.exists(embedding_path):
            try:
                # Load with memory mapping for better performance
                stored_embedding = np.load(embedding_path, mmap_mode='r')
                student_embeddings[student_folder] = stored_embedding.copy()
                student_count += 1
            except Exception as e:
                print(f"⚠ Error loading embedding for {student_folder}: {e}")

    print(f"✅ Loaded {student_count} student embeddings in total")



# ✅ Recognize a face using precomputed embeddings
# In face_recognition_model.py
def recognize_face(face_roi):
    """Optimized face recognition with caching."""
    try:
        # Resize to a smaller standard size
        face_roi = cv2.resize(face_roi, (112, 112))

        # Extract embedding using ArcFace
        face_embedding = DeepFace.represent(face_roi, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        face_embedding = np.array(face_embedding)
    except Exception as e:
        return "Unknown"

    if len(student_embeddings) == 0:
        load_student_embeddings()
        if len(student_embeddings) == 0:
            return "Unknown"

    # Pre-normalize face embedding
    normalized_face = face_embedding / np.linalg.norm(face_embedding)

    best_match = "Unknown"
    best_similarity = -1
    threshold = 0.65

    for student_folder, stored_embedding in student_embeddings.items():
        # Normalize stored embedding
        norm = np.linalg.norm(stored_embedding)
        if norm == 0:
            continue

        normalized_stored = stored_embedding / norm

        # Calculate cosine similarity
        similarity = np.dot(normalized_face, normalized_stored)

        if similarity > threshold and similarity > best_similarity:
            best_similarity = similarity
            parts = student_folder.split("_", 1)
            best_match = parts[1] if len(parts) > 1 else student_folder

    return best_match


# def update_attendance_file(index_number, name, date_str=None):
#     """
#     Comprehensive attendance file update with extensive logging
#     """
#     try:
#         # Ensure index_number and name are strings
#         index_number = str(index_number)
#         name = str(name)
#
#         # Use consistent date formatting
#         if date_str is None:
#             date_str = datetime.now().strftime("%Y-%m-%d")
#
#         # Verbose logging
#         print(f"🔍 Attempting to update attendance for:")
#         print(f"   Index Number: {index_number}")
#         print(f"   Name: {name}")
#         print(f"   Date: {date_str}")
#         print(f"   Attendance File: {os.path.abspath(ATTENDANCE_FILE)}")
#
#         # Ensure directory exists
#         os.makedirs(os.path.dirname(os.path.abspath(ATTENDANCE_FILE)), exist_ok=True)
#
#         # Create initial DataFrame if file doesn't exist
#         if not os.path.exists(ATTENDANCE_FILE):
#             print("📄 Creating new attendance file")
#             df = pd.DataFrame(columns=["Index No", "Name", "Date", "Time"])
#         else:
#             # Read existing file with error handling
#             try:
#                 df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
#                 print("📖 Successfully read existing attendance file")
#             except Exception as e:
#                 print(f"❌ Error reading file: {e}")
#                 # Create new DataFrame if reading fails
#                 df = pd.DataFrame(columns=["Index No", "Name", "Date", "Time"])
#
#         # Prepare new attendance entry
#         new_entry = {
#             "Index No": index_number,
#             "Name": name,
#             "Date": date_str,
#             "Time": datetime.now().strftime("%H:%M:%S")
#         }
#
#         # Check for duplicate entries on the same day
#         duplicate = df[
#             (df["Index No"] == index_number) &
#             (df["Name"] == name) &
#             (df["Date"] == date_str)
#         ]
#
#         if len(duplicate) == 0:
#             # Add new entry if no duplicate exists
#             df = df.append(new_entry, ignore_index=True)
#             print("✅ New attendance entry added")
#         else:
#             print("ℹ️ Attendance already marked for this student today")
#
#         # Save with comprehensive error handling
#         try:
#             df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
#             print(f"💾 Attendance file successfully updated at {ATTENDANCE_FILE}")
#         except PermissionError:
#             print("❌ ERROR: Cannot write to file. Is the file open?")
#         except Exception as e:
#             print(f"❌ Unexpected error saving file: {e}")
#
#         # Additional verification
#         print("\n📋 Current Attendance Data:")
#         print(df)
#
#     except Exception as e:
#         print(f"❌ CRITICAL ERROR in update_attendance_file: {e}")
#         import traceback
#         traceback.print_exc()



# ✅ Take Attendance Function (Optimized)
def take_attendance(video_capture=None, face_roi=None):
    """Detects and recognizes faces, updates attendance in a dynamic column-based format."""

    print("\n🔍 Starting Attendance Process")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Full Attendance File Path: {os.path.abspath(ATTENDANCE_FILE)}")

    recognized_names = []

    # Ensure the file exists before processing
    def ensure_attendance_file():
        """Create attendance file if it doesn't exist with a basic structure."""
        try:
            # Check if file exists
            if not os.path.exists(ATTENDANCE_FILE):
                # Create initial DataFrame
                initial_df = pd.DataFrame(columns=["Index No", "Name"])

                # Add today's date column
                today_date = datetime.now().strftime("%d-%m-%Y")
                initial_df[today_date] = ""

                # Save to Excel
                initial_df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
                print(f"✅ Created new attendance file: {ATTENDANCE_FILE}")
        except Exception as e:
            print(f"⚠ Error creating attendance file: {e}")

    # Call this to ensure file exists before processing
    ensure_attendance_file()

    recognized_names = []

    # Handle direct face_roi input case
    if face_roi is not None:
        name = recognize_face(face_roi)
        if name != "Unknown":
            index_number = None
            for student_folder in os.listdir(DATASET_PATH):
                if "_" in student_folder and student_folder.split("_", 1)[1] == name:
                    index_number = student_folder.split("_", 1)[0]
                    break

            if index_number:
                recognized_names.append((index_number, name))

    # Handle camera input case
    elif video_capture is not None:
        if not video_capture.isOpened():
            return "Error: Camera not available."

        ret, frame = video_capture.read()
        if not ret:
            return "Error: Could not access camera."

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector.detect_faces(rgb_frame)
        if not faces:
            return "⚠ No face detected."

        for face in faces:
            x, y, w, h = face["box"]

            # Ensure ROI is within valid image bounds
            height, width, _ = rgb_frame.shape
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)

            face_roi = rgb_frame[y:y + h, x:x + w]

            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                print(f"⚠ Skipping invalid face ROI")
                continue

            # Recognize face
            name = recognize_face(face_roi)

            if name != "Unknown":
                index_number = None
                for student_folder in os.listdir(DATASET_PATH):
                    if "_" in student_folder and student_folder.split("_", 1)[1] == name:
                        index_number = student_folder.split("_", 1)[0]
                        break

                if index_number:
                    recognized_names.append((index_number, name))
    else:
        return "Error: No input provided. Either video_capture or face_roi must be provided."

    # Update attendance for recognized faces
    if recognized_names:
        try:
            # Read the existing file
            df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")

            # Get today's date
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")

            # Ensure the date column exists
            if date_str not in df.columns:
                df[date_str] = ""

            updated_records = []

            for index_number, name in recognized_names:
                # Find or create the student record
                student_record = df[df["Index No"] == index_number]

                if len(student_record) == 0:
                    # Create new student record
                    new_record = pd.DataFrame({
                        "Index No": [index_number],
                        "Name": [name],
                        date_str: ["✅"]
                    })
                    df = pd.concat([df, new_record], ignore_index=True)
                else:
                    # Update existing student record
                    df.loc[df["Index No"] == index_number, date_str] = "✅"

                print(f"✅ Marked attendance for {name} ({index_number})")
                updated_records.append((index_number, name))

            # Sort columns to keep Index No and Name first, then date columns
            cols = ["Index No", "Name"] + sorted([col for col in df.columns if col not in ["Index No", "Name"]],
                                                 reverse=True)
            df = df[cols]

            # Verbose logging
            print(f"DataFrame before saving:\n{df}")

            # Save the updated DataFrame
            df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")

            return updated_records if updated_records else "✅ Already marked attendance for today."

        except Exception as e:
            print(f"⚠ Error updating attendance: {e}")
            import traceback
            traceback.print_exc()
            return f"Error updating attendance: {e}"

    return "⚠ No known face detected."


# Modify initialization to ensure file creation
def initialize_system():
    """Initialize the attendance system by creating necessary files."""
    try:
        # Ensure dataset directory exists
        os.makedirs(DATASET_PATH, exist_ok=True)

        # Create attendance file if it doesn't exist
        if not os.path.exists(ATTENDANCE_FILE):
            df = pd.DataFrame(columns=["Index No", "Name"])

            # Add today's date column
            today_date = datetime.now().strftime("%d-%m-%Y")
            df[today_date] = ""

            df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
            print(f"✅ Created attendance file: {ATTENDANCE_FILE}")

        # Load embeddings
        load_student_embeddings()
    except Exception as e:
        print(f"⚠ Error in system initialization: {e}")
        import traceback
        traceback.print_exc()





# ✅ Load embeddings when the program starts
load_student_embeddings()
initialize_system()