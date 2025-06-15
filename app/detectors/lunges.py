import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global state
reps = 0
stage = "up"  # assume user starts standing

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_lunges(image):
    global reps, stage

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "lunges", "form": "no_pose_detected", "reps": reps}

    landmarks = results.pose_landmarks.landmark

    # RIGHT leg only
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    angle = calculate_angle(hip, knee, ankle)

    # Classify form
    form = "mid"
    if 80 <= angle <= 100:
        form = "good"
    elif angle < 80:
        form = "too_deep"
    elif angle > 100:
        form = "not_deep_enough"
    if angle > 150:
        form = "standing"

    # Rep detection: down → up completes a rep
    if angle <= 100:
        if stage == "up":
            stage = "down"
            print("Down detected")  # Debug
    elif angle > 150:
        if stage == "down":
            reps += 1
            stage = "up"
            print("Rep counted")  # Debug

    return {
        "exercise": "lunges",
        "angle": angle,
        "form": form,
        "reps": reps
    }
