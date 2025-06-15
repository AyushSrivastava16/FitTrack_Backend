import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def analyze_jumping_jacks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "jumping_jacks", "form": "no_pose_detected", "reps": 0}

    landmarks = results.pose_landmarks.landmark

    # Get coordinates
    left_wrist = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
    ])
    right_wrist = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
    ])
    left_ankle = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
    ])
    right_ankle = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
    ])

    # Calculate distances
    hand_distance = calculate_distance(left_wrist, right_wrist)
    foot_distance = calculate_distance(left_ankle, right_ankle)

    # Thresholds (tune these as needed)
    hand_threshold = 0.5  # hands far apart = jumping
    foot_threshold = 0.5  # feet wide apart = jumping

    # Classify form
    form = "closed"
    if hand_distance > hand_threshold and foot_distance > foot_threshold:
        form = "open"
    elif hand_distance < 0.3 and foot_distance < 0.3:
        form = "closed"

    # NOTE: You can add repetition tracking based on transition from closed → open → closed

    return {
        "exercise": "jumping_jacks",
        "hand_distance": hand_distance,
        "foot_distance": foot_distance,
        "form": form,
        "reps": 0  # add logic to track state transitions for actual counting
    }
