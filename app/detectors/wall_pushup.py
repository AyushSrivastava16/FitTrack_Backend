import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# State tracking
reps = 0
stage = None  # 'up' or 'down'

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_wall_pushup(image):
    global reps, stage  # Use global to track state

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "wall_pushup", "form": "no_pose_detected", "reps": reps}

    landmarks = results.pose_landmarks.landmark

    # Use RIGHT arm
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # Calculate elbow angle
    angle = calculate_angle(shoulder, elbow, wrist)

    # Classify form
    form = "bad"
    if angle > 160:
        form = "up"
    elif angle < 90:
        form = "down"

    # Count reps
    if form == "down":
        stage = "down"
    if form == "up" and stage == "down":
        reps += 1
        stage = "up"

    # Debug output
    print(f"Form: {form}, Angle: {angle}, Reps: {reps}")

    return {
        "exercise": "wall_pushup",
        "angle": angle,
        "form": form,
        "reps": reps
    }
