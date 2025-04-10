#code of the GUI
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import face_recognition_model as model  # Import functions from face_recognition_model.py
from mtcnn import MTCNN
from tkinter import ttk  # Import Treeview widget
import threading
import queue
import time
from datetime import datetime
import numpy as np
import os
import threading
import time
import json
from datetime import datetime, date, timedelta
from PIL import Image, ImageTk










# Initialize Tkinter with enhanced styling
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("1920x1080")
root.configure(bg="#f0f2f5")  # Light blue-gray background

# Define color scheme
COLORS = {
    "primary": "#3b71ca",  # Main blue
    "secondary": "#9fa6b2",  # Light gray
    "success": "#14a44d",  # Green
    "danger": "#dc4c64",  # Red
    "warning": "#e4a11b",  # Yellow
    "info": "#54b4d3",  # Light blue
    "light": "#fbfbfb",  # Almost white
    "dark": "#332d2d"  # Dark gray
}

# Create styles for buttons and widgets
button_style = {
    "font": ("Arial", 12, "bold"),
    "bg": COLORS["primary"],
    "fg": "white",
    "activebackground": "#2b5592",  # Darker blue when clicked
    "activeforeground": "white",
    "relief": tk.RAISED,
    "borderwidth": 2,
    "padx": 15,
    "pady": 8,
    "cursor": "hand2"  # Hand cursor on hover
}

label_style = {
    "font": ("Arial", 12),
    "bg": COLORS["light"],
    "fg": COLORS["dark"],
    "padx": 5,
    "pady": 5
}



frame_style = {
    "bg": COLORS["light"],
    "relief": tk.RIDGE,
    "borderwidth": -1,
    "padx": 0,
    "pady": 0
}

# Initialize OpenCV Webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video_capture.set(cv2.CAP_PROP_FPS, 15)  # Set a reasonable FPS

# Initialize MTCNN for face detection
detector = MTCNN()

# Main container
container = tk.Frame(root, bg=COLORS["light"])
container.pack(fill="both", expand=True, padx=20, pady=20)

# Define camera mode
current_mode = "register"  # Track the active page
camera_running = False  # Control camera updates

# Define frames
main_frame = tk.Frame(container, **frame_style)
register_frame = tk.Frame(container, **frame_style)
attendance_frame = tk.Frame(container, **frame_style)



# Load background images (adjust paths and sizes as needed)
bg_main = ImageTk.PhotoImage(Image.open("Images/attendence.jpg").resize((1920,1080)))
bg_register = ImageTk.PhotoImage(Image.open("Images/attendence.jpg"))
bg_attendance = ImageTk.PhotoImage(Image.open("Images/attendence.jpg"))

main_bg_label = tk.Label(main_frame, image=bg_main)
main_bg_label.place(relwidth=1, relheight=1)  # Covers full frame

register_bg_label = tk.Label(register_frame, image=bg_register)
register_bg_label.place(relwidth=1, relheight=1)

attendance_bg_label = tk.Label(attendance_frame, image=bg_attendance)
attendance_bg_label.place(relwidth=1, relheight=1)


title_style = {
    "font": ("Arial", 18, "bold"),
    # "bg": main_bg_label.cget('bg'),
    "fg": COLORS["primary"],
    "padx": 10,
    "pady": 20,
    "highlightthickness": 0,
    "borderwidth": 0
}



for frame in (main_frame, register_frame, attendance_frame):
    frame.grid(row=0, column=0, sticky="nsew")

# Center content in container
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# # 📌 Main Page (Centered Content)
main_content = tk.Frame(main_frame, bg="#f0f2f5", bd=0, highlightthickness=0)
main_content.place(relx=0.5, rely=0.5, anchor="center")

main_title = tk.Label(main_bg_label, text="Face Recognition Attendance System", bg=main_bg_label.cget('bg'), **title_style)
main_title.place(relx=0.5, rely=0.10, anchor="center")

register_btn = tk.Button(main_bg_label, text="Register", command=lambda: switch_to_register(), **button_style)
register_btn.place(relx=0.5, rely=0.55, anchor="center")

attendance_btn = tk.Button(main_bg_label, text="Take Attendance", command=lambda: switch_to_attendance(), **button_style)
attendance_btn.place(relx=0.5, rely=0.65, anchor="center")



# Camera feed switching between registration and take attendance
def switch_to_attendance():
    global current_mode, camera_running, video_capture
    current_mode = "attendance"
    camera_running = True  # Allow camera updates

    # Reload the embeddings to include any newly registered students
    model.load_student_embeddings()

    # ✅ Ensure the camera is opened before updating frames
    if not video_capture.isOpened() or video_capture is None:
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        video_capture.set(cv2.CAP_PROP_FPS, 15)

    show_frame(attendance_frame)
    root.after(100, update_attendance_camera)  # ✅ Ensures the function is properly called


def switch_to_register():
    global current_mode, camera_running
    current_mode = "register"
    camera_running = True  # Allow camera updates
    show_frame(register_frame)
    update_camera()


# Stop camera updates
def stop_camera_and_return():
    global camera_running
    camera_running = False  # Stop camera updates
    show_frame(main_frame)


# 📌 Registration Page Layout
register_frame.grid_columnconfigure(0, weight=1)
register_frame.grid_columnconfigure(1, weight=1)
register_frame.grid_rowconfigure(0, weight=1)

# Left side: Camera Feed
camera_frame = tk.Frame(register_frame, bg=COLORS["dark"], padx=5, pady=5)
camera_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

camera_label = tk.Label(camera_frame, bg="black")
camera_label.grid(row=0, column=0, sticky="nsew")

# Right side: Form Input
form_frame = tk.Frame(register_frame, bg=COLORS["light"], padx=10, pady=10)
form_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

# Center the form content
form_frame.grid_rowconfigure(0, weight=1)
form_frame.grid_rowconfigure(6, weight=1)
form_frame.grid_columnconfigure(0, weight=1)
form_frame.grid_columnconfigure(3, weight=1)

reg_title = tk.Label(form_frame, text="Student Registration", **title_style)
reg_title.grid(row=1, column=1, columnspan=2, pady=10)

tk.Label(form_frame, text="Name:", **label_style).grid(row=2, column=1, sticky="w", pady=5)
entry_name = tk.Entry(form_frame, font=("Arial", 12), bd=2, relief=tk.GROOVE)
entry_name.grid(row=2, column=2, pady=5, padx=5, sticky="ew")

tk.Label(form_frame, text="Index No:", **label_style).grid(row=3, column=1, sticky="w", pady=5)
entry_index = tk.Entry(form_frame, font=("Arial", 12), bd=2, relief=tk.GROOVE)
entry_index.grid(row=3, column=2, pady=5, padx=5, sticky="ew")

register_btn = tk.Button(form_frame, text="Register", command=lambda: register(), **button_style)
register_btn.grid(row=4, column=1, columnspan=2, pady=15, sticky="ew")

back_btn = tk.Button(form_frame, text="Back", command=lambda: stop_camera_and_return(),
                     bg=COLORS["secondary"], fg="white",
                     activebackground="#8a8d93", activeforeground="white",
                     font=("Arial", 12), padx=15, pady=5, cursor="hand2")
back_btn.grid(row=5, column=1, columnspan=2, pady=10, sticky="ew")

# 📌 Attendance Page Layout
attendance_frame.grid_columnconfigure(0, weight=3, minsize=400)  # Set minimum width for Camera
attendance_frame.grid_columnconfigure(1, weight=1, minsize=200)  # Ensure Right Panel is visible

attendance_frame.grid_rowconfigure(0, weight=1)  # Camera (100% height)
attendance_frame.grid_rowconfigure(1, weight=1)  # Info Panel (Date & List)

# Left: Camera Feed with border
camera_frame = tk.Frame(attendance_frame, bg=COLORS["dark"], padx=5, pady=5)
camera_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

camera_label_attendance = tk.Label(camera_frame, bg="black")
camera_label_attendance.pack(fill="both", expand=True)

# Right: Date & Attended Students List
info_frame = tk.Frame(attendance_frame, bg=COLORS["light"], padx=5, pady=5)
info_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)

# Configure the style for Treeview
style = ttk.Style()
style.configure("Treeview",
                background=COLORS["light"],
                foreground=COLORS["dark"],
                rowheight=25,
                fieldbackground=COLORS["light"])
style.configure("Treeview.Heading",
                background=COLORS["primary"],
                foreground="white",
                font=('Arial', 10, 'bold'))
style.map('Treeview', background=[('selected', COLORS["info"])])

date_label = tk.Label(info_frame, text=f"Date: {datetime.now().strftime('%Y-%m-%d')}",
                      font=("Arial", 14, "bold"), bg=COLORS["light"], fg=COLORS["primary"])
date_label.grid(row=0, column=0, padx=10, pady=5, sticky="nw")

attendance_tree = ttk.Treeview(info_frame, columns=("Name", "Time"), show="headings", height=15)
attendance_tree.heading("Name", text="Name")
attendance_tree.heading("Time", text="Time")
attendance_tree.column("Name", width=100, anchor="w")
attendance_tree.column("Time", width=100, anchor="center")

attendance_tree.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
info_frame.grid_rowconfigure(1, weight=1)

# Add scrollbar to treeview
scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=attendance_tree.yview)
scrollbar.grid(row=1, column=1, sticky="ns")
attendance_tree.configure(yscrollcommand=scrollbar.set)

back_btn = tk.Button(info_frame, text="Back", command=lambda: stop_camera_and_return(),
                     bg=COLORS["secondary"], fg="white",
                     activebackground="#8a8d93", activeforeground="white",
                     font=("Arial", 12), padx=15, pady=5, cursor="hand2")
back_btn.grid(row=2, column=0, padx=10, pady=10, sticky="se")


# 📌 Show Frame Function
def show_frame(frame):
    frame.tkraise()


frame_count = 0  # Counter to skip frames
face_queue = queue.Queue()
face_detection_running = False  # Prevent multiple face detection threads


def detect_faces_async(frame):
    """Threaded function to detect faces and store only the latest results in a queue."""
    global face_detection_running
    detected = detector.detect_faces(frame)

    # ✅ Clear old faces & store only the latest detected faces
    while not face_queue.empty():
        face_queue.get()

    face_queue.put(detected)  # Store latest detected faces
    face_detection_running = False  # Allow next detection


def update_camera():
    """Efficiently updates the camera feed with reduced lag and error handling."""
    global frame_count, camera_running, face_detection_running, video_capture

    if current_mode != "register" or not camera_running:
        return  # ✅ Stop updating if not in registration mode

    if not video_capture.isOpened():  # ✅ Reopen camera if it was closed
        print("⚠ Camera was closed. Reopening...")
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_count += 1

    # ✅ Skip 2 out of 3 frames for better performance
    if frame_count % 3 != 0:
        camera_label.after(10, update_camera)
        return

    ret, frame = video_capture.read()
    if not ret:
        print("⚠ Error: Could not read frame from camera. Retrying...")
        camera_label.after(10, update_camera)  # ✅ Retry instead of stopping
        return

    frame = cv2.flip(frame, 1)  # ✅ Flip for a mirror effect
    frame = cv2.resize(frame, (320, 240))  # ✅ Reduced resolution for efficiency
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ Run face detection in a separate thread only if it's not already running
    if not face_detection_running:
        face_detection_running = True
        threading.Thread(target=detect_faces_async, args=(rgb_frame,), daemon=True).start()

    # ✅ Retrieve the latest detected faces from the queue
    detected_faces = []
    if not face_queue.empty():
        detected_faces = face_queue.get()

    # ✅ Draw bounding boxes around detected faces
    for face in detected_faces:
        x, y, w, h = face["box"]
        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ✅ Convert to Tkinter-compatible image and update the UI
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    # ✅ Ensure UI updates smoothly
    camera_label.after(10, update_camera)
    cv2.waitKey(1)  # ✅ Helps with rendering smoothness


frame_count_attendance = 0  # Counter to skip frames
face_queue_attendance = queue.Queue()
face_detection_running_attendance = False  # Prevent multiple detection threads


def detect_faces_attendance_async(frame):
    global face_detection_running_attendance
    detected = detector.detect_faces(frame)

    # ✅ If no faces are detected, retry detection on next frame
    if not detected:
        face_detection_running_attendance = False
        return

    # ✅ Clear old queue & store latest detection
    while not face_queue_attendance.empty():
        face_queue_attendance.get_nowait()

    face_queue_attendance.put(detected)
    face_detection_running_attendance = False  # Allow next detection


# Create a global flag to control face recognition frequency
last_recognition_time = 0
recognition_cooldown = 0.01  # Only recognize faces every 1.5 seconds


def update_attendance_camera():
    """Optimized camera feed with reduced lag during attendance taking."""
    global frame_count_attendance, camera_running, face_detection_running_attendance, last_recognition_time

    if current_mode != "attendance" or not camera_running:
        return

    # Skip frames for better performance (process every 4th frame)
    frame_count_attendance += 1
    if frame_count_attendance % 4 != 0:
        camera_label_attendance.after(10, update_attendance_camera)
        return

    # Read frame
    ret, frame = video_capture.read()
    if not ret:
        camera_label_attendance.after(10, update_attendance_camera)
        return

    # Create a copy of the frame for processing
    display_frame = cv2.flip(frame, 1)
    processing_frame = cv2.resize(display_frame.copy(), (160, 120))  # Process at lower resolution

    # Start face detection in separate thread only when needed
    current_time = time.time()
    perform_recognition = (current_time - last_recognition_time >= recognition_cooldown)

    if perform_recognition and not face_detection_running_attendance:
        face_detection_running_attendance = True
        last_recognition_time = current_time

        # Create a separate thread to handle face detection and recognition
        threading.Thread(
            target=process_face_for_attendance,
            args=(processing_frame, display_frame),
            daemon=True
        ).start()

    # Convert display frame for Tkinter
    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(display_frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label_attendance.imgtk = imgtk
    camera_label_attendance.configure(image=imgtk)

    # Continue updating camera
    camera_label_attendance.after(10, update_attendance_camera)


def process_face_for_attendance(small_frame, full_frame):
    """Process faces in a separate thread."""
    global face_detection_running_attendance

    # Convert to RGB for processing
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces at low resolution
    faces = detector.detect_faces(rgb_small_frame)

    if faces:
        # Scale coordinates to match the display frame
        scale_x = full_frame.shape[1] / small_frame.shape[1]
        scale_y = full_frame.shape[0] / small_frame.shape[0]

        processed_results = []

        for face in faces:
            # Scale face coordinates
            x, y, w, h = face["box"]
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)

            # Extract face ROI from full frame
            face_roi = full_frame[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w]

            if face_roi.size == 0:
                continue

            # Run face recognition
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            name = model.recognize_face(face_rgb)
            model.take_attendance(face_roi=face_rgb)
            processed_results.append((scaled_x, scaled_y, scaled_w, scaled_h, name))

        # Update UI with recognition results
        root.after(0, lambda: update_recognition_results(full_frame, processed_results))

    face_detection_running_attendance = False


def update_recognition_results(frame, results):
    """Update UI with face recognition results."""
    # Make a copy of the frame
    annotated_frame = frame.copy()

    for x, y, w, h, name in results:
        # Draw bounding boxes and labels
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # If a known person is detected, mark attendance
        if name != "Unknown":
            mark_attendance_for_person(name)

    # Convert to RGB for display
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label_attendance.imgtk = imgtk
    camera_label_attendance.configure(image=imgtk)


def mark_attendance_for_person(name):
    """Mark attendance for a recognized person."""
    # Find the student's index number
    index_number = None
    for student_folder in os.listdir(model.DATASET_PATH):
        if "_" in student_folder and student_folder.split("_", 1)[1] == name:
            index_number = student_folder.split("_", 1)[0]
            break

    if index_number:
        # Use a thread to handle attendance marking
        threading.Thread(
            target=lambda: process_attendance_marking(index_number, name),
            daemon=True
        ).start()


def process_attendance_marking(index_number, name):
    """Process attendance marking in a separate thread."""
    result = model.update_attendance_file(index_number, name)

    # If attendance was marked successfully, update the UI
    if isinstance(result, list):
        for entry in result:
            if entry[0] == index_number:
                root.after(0, lambda: attendance_tree.insert("", "end", values=(name, entry[2])))


# 📌 Register Function
def register():
    name = entry_name.get().strip()
    index_number = entry_index.get().strip()

    if not name or not index_number:
        messagebox.showerror("Error", "Please enter Name and Index Number")
        return

    # Create a new window to display the registration process
    reg_window = tk.Toplevel(root)
    reg_window.title("Student Registration")
    reg_window.geometry("640x540")
    reg_window.configure(bg=COLORS["light"])

    # Add labels to display instructions and progress
    instruction_label = tk.Label(reg_window, text="Preparing...",
                                 font=("Arial", 14, "bold"), bg=COLORS["light"], fg=COLORS["primary"])
    instruction_label.pack(pady=10)

    progress_label = tk.Label(reg_window, text="Progress: 0/30",
                              font=("Arial", 12), bg=COLORS["light"], fg=COLORS["dark"])
    progress_label.pack(pady=5)

    countdown_label = tk.Label(reg_window, text="",
                               font=("Arial", 18, "bold"), bg=COLORS["light"], fg=COLORS["warning"])
    countdown_label.pack(pady=5)

    # Camera display in registration window
    reg_camera_frame = tk.LabelFrame(reg_window, text="Camera", bg=COLORS["dark"], fg="white",
                                     padx=5, pady=5, font=("Arial", 10, "bold"))
    reg_camera_frame.pack(pady=10, fill="both", expand=True)

    reg_camera_label = tk.Label(reg_camera_frame, bg="black")
    reg_camera_label.pack(pady=5, fill="both", expand=True)

    # Status label
    status_label = tk.Label(reg_window, text="Getting ready...",
                            font=("Arial", 12), bg=COLORS["light"], fg=COLORS["info"])
    status_label.pack(pady=5)

    # Define poses
    poses = [
        {"name": "front", "count": 0, "target": 10, "instruction": "Look directly at the camera"},
        {"name": "right", "count": 0, "target": 10, "instruction": "Turn your head to the RIGHT"},
        {"name": "left", "count": 0, "target": 10, "instruction": "Turn your head to the LEFT"}
    ]

    # Registration state variables
    reg_state = {
        "current_pose_index": 0,
        "countdown_start": None,
        "countdown_value": 3,
        "is_countdown": True,
        "last_capture_time": 0,
        "capture_interval": 0.3,
        "embeddings": [],
        "student_folder": os.path.join(model.DATASET_PATH, f"{index_number}_{name}"),
        "completed": False
    }

    # Create student folder
    os.makedirs(reg_state["student_folder"], exist_ok=True)

    def update_registration_feed():
        if reg_state["completed"]:
            reg_window.destroy()
            return

        # Get current pose
        current_pose_index = reg_state["current_pose_index"]
        if current_pose_index >= len(poses):
            # Save embeddings and finish
            if reg_state["embeddings"]:
                mean_embedding = np.mean(reg_state["embeddings"], axis=0)
                np.save(os.path.join(reg_state["student_folder"], "embedding.npy"), mean_embedding)
                status_label.config(text=f"✅ Registration completed successfully!")

                # Close the window after a delay
                reg_window.after(2000, reg_window.destroy)
                reg_state["completed"] = True

                # Show success message in main window
                messagebox.showinfo("Registration", f"✅ Registration Completed: {name} ({index_number})")

                # Clear entry fields
                entry_name.delete(0, tk.END)
                entry_index.delete(0, tk.END)
                return

        current_pose = poses[current_pose_index]

        # Update instruction and progress labels
        instruction_label.config(text=current_pose["instruction"])
        progress_label.config(text=f"Progress: {current_pose['count']}/{current_pose['target']} "
                                   f"(Pose {current_pose_index + 1}/3)")

        # Capture frame
        ret, frame = video_capture.read()
        if not ret:
            status_label.config(text="⚠ Camera error. Please try again.")
            reg_window.after(10, update_registration_feed)
            return

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        faces = model.detector.detect_faces(rgb_frame)

        # Process faces if detected
        is_correct_pose = False
        if faces:
            face = faces[0]
            x, y, w, h = face['box']

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check face orientation
            landmarks = face['keypoints']
            is_correct_pose = model.check_face_orientation(landmarks, current_pose["name"])

            # Draw orientation status
            orientation_text = "Correct Pose ✓" if is_correct_pose else "Adjust Position ✗"
            color = (0, 255, 0) if is_correct_pose else (0, 0, 255)
            cv2.putText(frame, orientation_text, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Handle countdown and capturing
        current_time = time.time()

        if reg_state["is_countdown"]:
            # Countdown mode
            if is_correct_pose:
                if reg_state["countdown_start"] is None:
                    reg_state["countdown_start"] = current_time

                elapsed = current_time - reg_state["countdown_start"]
                remaining = max(0, reg_state["countdown_value"] - int(elapsed))

                # Show countdown
                countdown_label.config(text=f"Starting in: {remaining}")

                # When countdown finishes
                if remaining == 0:
                    reg_state["is_countdown"] = False
                    reg_state["last_capture_time"] = current_time
                    countdown_label.config(text="Capturing...")
            else:
                # Reset countdown if pose changes
                reg_state["countdown_start"] = None
                countdown_label.config(text="Position your face correctly")
        else:
            # Capture mode
            if is_correct_pose and (current_time - reg_state["last_capture_time"] >= reg_state["capture_interval"]):
                # Extract face region
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                cropped_face = rgb_frame[y:y + h, x:x + w]

                if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                    # Resize for consistency
                    cropped_face = cv2.resize(cropped_face, (112, 112))

                    # Save image
                    img_path = os.path.join(reg_state["student_folder"],
                                            f"{current_pose['name']}_{current_pose['count'] + 1}.jpg")
                    cv2.imwrite(img_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

                    # Extract embedding
                    try:
                        embedding = \
                        model.DeepFace.represent(cropped_face, model_name="ArcFace", enforce_detection=False)[0][
                            "embedding"]
                        reg_state["embeddings"].append(embedding)

                        # Update counters
                        poses[current_pose_index]["count"] += 1
                        current_pose["count"] = poses[current_pose_index]["count"]
                        reg_state["last_capture_time"] = current_time

                        # Flash indicator on screen
                        cv2.putText(frame, "Captured!", (frame.shape[1] - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Update progress
                        progress_label.config(text=f"Progress: {current_pose['count']}/{current_pose['target']} "
                                                   f"(Pose {current_pose_index + 1}/3)")
                    except Exception as e:
                        print(f"⚠ Could not extract embedding: {e}")

            # Check if we've completed the current pose
            if poses[current_pose_index]["count"] >= current_pose["target"]:
                reg_state["current_pose_index"] += 1
                # Reset to countdown mode for next pose
                reg_state["is_countdown"] = True
                reg_state["countdown_start"] = None
                countdown_label.config(text="Getting ready for next pose...")

                if reg_state["current_pose_index"] < len(poses):
                    status_label.config(text=f"Moving to next pose: {poses[reg_state['current_pose_index']]['name']}")
                else:
                    status_label.config(text="Processing registration data...")

        # Convert to Tkinter-compatible image and update display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        reg_camera_label.imgtk = imgtk
        reg_camera_label.configure(image=imgtk)

        # Continue updating
        reg_window.after(10, update_registration_feed)

    # Start the update process
    update_registration_feed()


# 📌 Take Attendance Function
def take_attendance(index_number, name):
    attendance_data = model.update_attendance_file(index_number, name)

    if isinstance(attendance_data, list):
        for entry in attendance_data:
            # ✅ Append new records instead of clearing everything
            attendance_tree.insert("", "end", values=(entry[1], entry[2]))
    else:
        messagebox.showinfo("Attendance", attendance_data)


# Add a new attendance tracking mechanism
def load_daily_attendance():
    """Load or create daily attendance record."""
    today = date.today().strftime("%Y-%m-%d")
    attendance_file = os.path.join(model.DATASET_PATH, f"attendance_{today}.json")

    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            return json.load(f)
    return {}


def save_daily_attendance(attendance_record):
    """Save daily attendance record."""
    today = date.today().strftime("%Y-%m-%d")
    attendance_file = os.path.join(model.DATASET_PATH, f"attendance_{today}.json")

    with open(attendance_file, 'w') as f:
        json.dump(attendance_record, f, indent=4)


# Global variable to track last attendance times
last_attendance_times = {}
ATTENDANCE_COOLDOWN = 300  # 5 minutes cooldown between attendance marks


def mark_attendance_for_person(name):
    """Mark attendance for a recognized person with cooldown."""
    global last_attendance_times

    # Find the student's index number
    index_number = None
    for student_folder in os.listdir(model.DATASET_PATH):
        if "_" in student_folder and student_folder.split("_", 1)[1] == name:
            index_number = student_folder.split("_", 1)[0]
            break

    if not index_number:
        return

    current_time = time.time()

    # Check if student has already been marked for attendance today
    daily_attendance = load_daily_attendance()
    today = date.today().strftime("%Y-%m-%d")

    # Check cooldown and prevent duplicate marking
    if (index_number not in daily_attendance.get(today, {}) and
            (index_number not in last_attendance_times or
             current_time - last_attendance_times.get(index_number, 0) > ATTENDANCE_COOLDOWN)):

        # Mark attendance
        current_datetime = datetime.now()
        attendance_time = current_datetime.strftime("%H:%M:%S")

        # Update daily attendance record
        if today not in daily_attendance:
            daily_attendance[today] = {}

        daily_attendance[today][index_number] = {
            "name": name,
            "time": attendance_time
        }

        # Save updated attendance
        save_daily_attendance(daily_attendance)

        # Update last attendance time
        last_attendance_times[index_number] = current_time

        # Update UI (thread-safe)
        root.after(0, lambda: update_attendance_treeview(name, attendance_time))


def update_attendance_treeview(name, attendance_time):
    """Update attendance treeview, preventing duplicates."""
    # Check if the entry already exists
    existing_entries = attendance_tree.get_children()
    for entry in existing_entries:
        values = attendance_tree.item(entry, "values")
        if values and values[0] == name and values[1] == attendance_time:
            return  # Already exists, don't add duplicate

    # Insert new entry if not a duplicate
    attendance_tree.insert("", "end", values=(name, attendance_time))


def clear_daily_attendance():
    """Clear attendance records at midnight."""
    while True:
        # Sleep until midnight
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        time_to_wait = (tomorrow - now).total_seconds()
        time.sleep(time_to_wait)

        # Clear last attendance times and reset
        last_attendance_times.clear()

        # Optional: You could add logging or notifications here


# Start a background thread to manage daily attendance clearing
attendance_clearing_thread = threading.Thread(target=clear_daily_attendance, daemon=True)
attendance_clearing_thread.start()


# Start the camera update
update_camera()
show_frame(main_frame)
root.mainloop()
