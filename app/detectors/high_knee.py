import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Rep tracking
reps = 0
stage = None  # "up" or "down"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_high_knee(image):
    global reps, stage

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "high_knee", "form": "no_pose_detected", "reps": reps}

    landmarks = results.pose_landmarks.landmark

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    # Calculate angles
    left_angle = calculate_angle(left_hip, left_knee, [left_knee[0], left_knee[1] + 0.1])
    right_angle = calculate_angle(right_hip, right_knee, [right_knee[0], right_knee[1] + 0.1])

    # Form classification
    form = "none"
    if left_angle > 80:
        form = "left_knee_up"
    elif right_angle > 80:
        form = "right_knee_up"

    # Rep counting logic
    if (left_angle > 80 or right_angle > 80):
        if stage == "down":
            reps += 1
            stage = "up"
    elif (left_angle < 45 and right_angle < 45):
        stage = "down"

    return {
        "exercise": "high_knee",
        "form": form,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "reps": reps
    }
