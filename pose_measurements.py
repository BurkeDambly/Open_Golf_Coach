import math

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None


MIN_LANDMARK_VISIBILITY = 0.5
PANEL_WIDTH = 450
PANEL_HEIGHT = 275
WINDOW_NAME = "Open Golf Coach - Head-On Swing Analyzer"
SMOOTHING_ALPHA = 0.35
MODEL_PANEL_WIDTH = 280

TRACKED_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

CLEAN_POSE_SEGMENTS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


def _distance_between_points(point_a, point_b):
    """Calculate the straight-line pixel distance between two [x, y] points."""
    x_difference = point_a[0] - point_b[0]
    y_difference = point_a[1] - point_b[1]
    return math.sqrt((x_difference * x_difference) + (y_difference * y_difference))


def _average(values):
    """Return the average of a list, or None when no values were collected."""
    if not values:
        return None

    return sum(values) / len(values)


def _minimum(values):
    """Return the minimum of a list, or None when no values were collected."""
    if not values:
        return None

    return min(values)


def _maximum(values):
    """Return the maximum of a list, or None when no values were collected."""
    if not values:
        return None

    return max(values)


def _movement_range(points):
    """Estimate total movement as the diagonal range of tracked point positions."""
    if not points:
        return None

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return _distance_between_points(
        [min(x_values), min(y_values)],
        [max(x_values), max(y_values)],
    )


def _horizontal_range(points):
    """Return the left-to-right movement range for a tracked point."""
    if not points:
        return None

    x_values = [point[0] for point in points]
    return max(x_values) - min(x_values)


def _landmark_to_pixel(landmarks, landmark_name, frame_width, frame_height):
    """Convert a visible MediaPipe landmark into [x, y] pixel coordinates."""
    pose_landmark = mp.solutions.pose.PoseLandmark[landmark_name]
    landmark = landmarks[pose_landmark.value]

    if landmark.visibility < MIN_LANDMARK_VISIBILITY:
        return None

    return [int(landmark.x * frame_width), int(landmark.y * frame_height)]


def _landmark_to_world(landmarks, landmark_name):
    """Convert a MediaPipe world landmark into [x, y, z] coordinates."""
    pose_landmark = mp.solutions.pose.PoseLandmark[landmark_name]
    landmark = landmarks[pose_landmark.value]

    if landmark.visibility < MIN_LANDMARK_VISIBILITY:
        return None

    return [landmark.x, landmark.y, landmark.z]


def _midpoint(point_a, point_b):
    """Find the midpoint between two [x, y] points."""
    return [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2]


def _line_angle_degrees(point_a, point_b):
    """Calculate the angle of a line between two points in image space."""
    x_difference = point_b[0] - point_a[0]
    y_difference = point_b[1] - point_a[1]
    return math.degrees(math.atan2(y_difference, x_difference))


def _angle_from_horizontal(point_a, point_b):
    """Calculate a readable tilt angle against the horizontal axis."""
    angle = _line_angle_degrees(point_a, point_b)

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def _angle_from_vertical(bottom_point, top_point):
    """Calculate body lean against vertical; positive means screen-right lean."""
    x_difference = top_point[0] - bottom_point[0]
    y_difference = bottom_point[1] - top_point[1]
    return math.degrees(math.atan2(x_difference, y_difference))


def _format_metric(value, suffix=""):
    """Format a metric for the live video panel."""
    if value is None:
        return "--"

    return f"{value:.1f}{suffix}"


def _draw_text(frame, text, position, scale=0.55, color=(255, 255, 255), thickness=1):
    """Draw readable text on the OpenCV frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_clean_pose(frame, points):
    """Draw a simple golfer model instead of the full MediaPipe skeleton."""
    line_color = (0, 210, 255)
    joint_color = (60, 255, 60)

    for start_name, end_name in CLEAN_POSE_SEGMENTS:
        start_point = points.get(start_name)
        end_point = points.get(end_name)

        if start_point and end_point:
            cv2.line(
                frame,
                tuple(map(int, start_point)),
                tuple(map(int, end_point)),
                line_color,
                3,
            )

    for point in points.values():
        if point:
            cv2.circle(frame, tuple(map(int, point)), 5, joint_color, -1)


def _draw_live_dashboard(frame, frame_number, pose_detected, metrics, paused):
    """Draw a compact dashboard with the most useful head-on swing numbers."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (PANEL_WIDTH, PANEL_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status = "LOCKED" if pose_detected else "SEARCHING"
    status_color = (80, 255, 80) if pose_detected else (60, 180, 255)
    playback_status = "PAUSED" if paused else "PLAYING"

    _draw_text(frame, "Open Golf Coach", (25, 38), 0.75, (255, 255, 255), 2)
    _draw_text(frame, f"Pose: {status}", (25, 68), 0.6, status_color, 2)
    _draw_text(frame, f"Frame: {frame_number}", (255, 68), 0.6, (220, 220, 220), 1)
    _draw_text(frame, f"Playback: {playback_status}", (25, 94), 0.55, (220, 220, 220), 1)

    rows = [
        ("Shoulder tilt", _format_metric(metrics.get("shoulder_tilt_degrees"), " deg")),
        ("Hip tilt", _format_metric(metrics.get("hip_tilt_degrees"), " deg")),
        ("Torso lean", _format_metric(metrics.get("torso_lean_degrees"), " deg")),
        ("Head sway", _format_metric(metrics.get("head_sway_pixels"), " px")),
        ("Hip sway", _format_metric(metrics.get("hip_sway_pixels"), " px")),
        ("Hand path", _format_metric(metrics.get("hand_path_pixels"), " px")),
    ]

    y_position = 130
    for label, value in rows:
        _draw_text(frame, label, (25, y_position), 0.55, (190, 190, 190), 1)
        _draw_text(frame, value, (250, y_position), 0.55, (255, 255, 255), 1)
        y_position += 24

    _draw_text(
        frame,
        "Space/p: pause | Trackbar: scrub | q: quit",
        (25, 258),
        0.5,
        (170, 170, 170),
        1,
    )


def _empty_trackbar_callback(position):
    """OpenCV trackbars require a callback, even when the loop reads the value."""
    return None


def _smooth_points(current_points, previous_points, alpha=SMOOTHING_ALPHA):
    """Smooth landmark movement to reduce visual jitter and metric spikes."""
    smoothed_points = {}

    for landmark_name, current_point in current_points.items():
        previous_point = previous_points.get(landmark_name)

        if current_point is None:
            smoothed_points[landmark_name] = previous_point
        elif previous_point is None:
            smoothed_points[landmark_name] = current_point
        else:
            smoothed_points[landmark_name] = [
                (alpha * current_point[index]) + ((1 - alpha) * previous_point[index])
                for index in range(len(current_point))
            ]

    return smoothed_points


def _draw_world_pose_panel(frame, world_points):
    """Draw a pseudo-3D body model from MediaPipe world landmarks."""
    if not world_points:
        return

    frame_height, frame_width = frame.shape[:2]
    panel_left = max(frame_width - MODEL_PANEL_WIDTH - 10, PANEL_WIDTH + 20)
    panel_right = frame_width - 10
    panel_top = 10
    panel_bottom = min(330, frame_height - 10)

    if panel_left >= panel_right or panel_top >= panel_bottom:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    _draw_text(frame, "3D pose estimate", (panel_left + 15, panel_top + 28), 0.55, (255, 255, 255), 1)
    _draw_text(frame, "front + depth", (panel_left + 15, panel_top + 52), 0.45, (180, 180, 180), 1)

    projected_points = {}
    center_x = (panel_left + panel_right) / 2
    center_y = (panel_top + panel_bottom) / 2 + 35
    scale = min(panel_right - panel_left, panel_bottom - panel_top) * 0.85

    for landmark_name, point in world_points.items():
        if point is None:
            continue

        # Mix x and z so the stick figure reads as a simple 3D model.
        projected_x = center_x + ((point[0] - point[2] * 0.45) * scale)
        projected_y = center_y + (point[1] * scale)
        projected_points[landmark_name] = [projected_x, projected_y]

    for start_name, end_name in CLEAN_POSE_SEGMENTS:
        start_point = projected_points.get(start_name)
        end_point = projected_points.get(end_name)

        if start_point and end_point:
            cv2.line(
                frame,
                tuple(map(int, start_point)),
                tuple(map(int, end_point)),
                (255, 180, 80),
                2,
            )

    for point in projected_points.values():
        cv2.circle(frame, tuple(map(int, point)), 4, (80, 220, 255), -1)


def _process_frame_with_pose(
    frame,
    pose,
    frame_number,
    metric_history,
    head_positions,
    hip_center_positions,
    hand_center_positions,
    previous_points,
    previous_world_points,
):
    """Run pose detection for one frame and draw the clean overlay."""
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    if not pose_results.pose_landmarks:
        return False, {}, previous_points, previous_world_points

    landmarks = pose_results.pose_landmarks.landmark
    points = _extract_pose_points(landmarks, frame_width, frame_height)
    points = _smooth_points(points, previous_points)
    current_metrics = _calculate_frame_measurements(points)

    _record_frame_measurements(
        current_metrics,
        metric_history,
        head_positions,
        hip_center_positions,
        hand_center_positions,
    )
    current_metrics["head_sway_pixels"] = _horizontal_range(head_positions)
    current_metrics["hip_sway_pixels"] = _horizontal_range(hip_center_positions)
    current_metrics["hand_path_pixels"] = _movement_range(hand_center_positions)

    world_points = previous_world_points
    if pose_results.pose_world_landmarks:
        world_points = _extract_world_points(pose_results.pose_world_landmarks.landmark)
        world_points = _smooth_points(world_points, previous_world_points)

    _draw_clean_pose(frame, points)
    _draw_world_pose_panel(frame, world_points)
    return True, current_metrics, points, world_points


def analyze_head_on_swing(video_capture):
    """Analyze one head-on golf swing video using MediaPipe body keypoints."""
    if mp is None or not hasattr(mp, "solutions"):
        print("MediaPipe is not installed correctly.")
        print("Run: python -m pip install --force-reinstall mediapipe==0.10.21")
        return _empty_analysis_results()

    mp_pose = mp.solutions.pose

    frame_number = 0
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_frame_count = 0
    missed_frame_count = 0
    paused = False
    last_raw_frame = None
    last_pose_detected = False
    last_metrics = {}
    last_trackbar_position = 0
    previous_points = {}
    previous_world_points = {}

    metric_history = {
        "stance_width_pixels": [],
        "shoulder_tilt_degrees": [],
        "hip_tilt_degrees": [],
        "torso_lean_degrees": [],
    }
    head_positions = []
    hip_center_positions = []
    hand_center_positions = []

    print("\nAnalyzing swing.")
    print("Controls: spacebar or 'p' pauses, the trackbar scrubs, and 'q' quits.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar(
        "Frame",
        WINDOW_NAME,
        0,
        max(frame_count - 1, 1),
        _empty_trackbar_callback,
    )

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            trackbar_position = cv2.getTrackbarPos("Frame", WINDOW_NAME)
            user_scrubbed = abs(trackbar_position - last_trackbar_position) > 1

            if user_scrubbed:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, trackbar_position)
                previous_points = {}
                previous_world_points = {}
                last_raw_frame = None

            if paused and last_raw_frame is not None and not user_scrubbed:
                frame = last_raw_frame.copy()
                _draw_live_dashboard(
                    frame,
                    frame_number,
                    last_pose_detected,
                    last_metrics,
                    paused,
                )
                cv2.imshow(WINDOW_NAME, frame)
            else:
                success, frame = video_capture.read()

                if not success:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cv2.setTrackbarPos("Frame", WINDOW_NAME, 0)
                    last_trackbar_position = 0
                    previous_points = {}
                    previous_world_points = {}
                    last_raw_frame = None
                    continue

                frame_number = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                (
                    pose_detected,
                    current_metrics,
                    previous_points,
                    previous_world_points,
                ) = _process_frame_with_pose(
                    frame,
                    pose,
                    frame_number,
                    metric_history,
                    head_positions,
                    hip_center_positions,
                    hand_center_positions,
                    previous_points,
                    previous_world_points,
                )

                if pose_detected:
                    detected_frame_count += 1
                else:
                    missed_frame_count += 1

                last_raw_frame = frame.copy()
                last_pose_detected = pose_detected
                last_metrics = current_metrics.copy()
                _draw_live_dashboard(
                    frame,
                    frame_number,
                    pose_detected,
                    current_metrics,
                    paused,
                )
                cv2.imshow(WINDOW_NAME, frame)

                if frame_count > 0:
                    current_position = min(frame_number, frame_count - 1)
                    cv2.setTrackbarPos("Frame", WINDOW_NAME, current_position)
                    last_trackbar_position = current_position

            key = cv2.waitKeyEx(30)

            if key in (ord("q"), ord("Q")):
                break
            if key in (ord(" "), ord("p"), ord("P")):
                paused = not paused

    cv2.destroyAllWindows()

    return _build_analysis_summary(
        frame_number,
        detected_frame_count,
        missed_frame_count,
        metric_history,
        head_positions,
        hip_center_positions,
        hand_center_positions,
    )


def _empty_analysis_results():
    """Return the summary shape when analysis cannot run."""
    return {
        "detection_quality": {
            "frames_processed": 0,
            "pose_detected_frames": 0,
            "pose_missed_frames": 0,
            "pose_detection_rate_percent": None,
        },
        "setup": {
            "average_stance_width_pixels": None,
        },
        "body_angles": {
            "average_shoulder_tilt_degrees": None,
            "maximum_shoulder_tilt_degrees": None,
            "average_hip_tilt_degrees": None,
            "maximum_hip_tilt_degrees": None,
            "average_torso_lean_degrees": None,
            "maximum_torso_lean_degrees": None,
        },
        "movement": {
            "head_sway_pixels": None,
            "hip_sway_pixels": None,
            "hand_path_pixels": None,
        },
        "notes": [
            "Front-view 2D video is useful for tilt and lateral sway.",
            "True shoulder turn, hip turn, X-factor, and weight shift need 3D data or another camera view.",
        ],
    }


def _extract_pose_points(landmarks, frame_width, frame_height):
    """Extract the body points this analyzer cares about."""
    points = {}

    for landmark_name in TRACKED_LANDMARKS:
        points[landmark_name] = _landmark_to_pixel(
            landmarks,
            landmark_name,
            frame_width,
            frame_height,
        )

    return points


def _extract_world_points(landmarks):
    """Extract 3D world points for the pseudo-3D model panel."""
    points = {}

    for landmark_name in TRACKED_LANDMARKS:
        points[landmark_name] = _landmark_to_world(landmarks, landmark_name)

    return points


def _calculate_frame_measurements(points):
    """Calculate a focused set of useful head-on swing measurements."""
    left_shoulder = points.get("LEFT_SHOULDER")
    right_shoulder = points.get("RIGHT_SHOULDER")
    left_hip = points.get("LEFT_HIP")
    right_hip = points.get("RIGHT_HIP")
    left_ankle = points.get("LEFT_ANKLE")
    right_ankle = points.get("RIGHT_ANKLE")
    left_wrist = points.get("LEFT_WRIST")
    right_wrist = points.get("RIGHT_WRIST")
    nose = points.get("NOSE")

    measurements = {
        "stance_width_pixels": None,
        "shoulder_tilt_degrees": None,
        "hip_tilt_degrees": None,
        "torso_lean_degrees": None,
        "head_position": nose,
        "hip_center": None,
        "hand_center": None,
    }

    if left_ankle and right_ankle:
        measurements["stance_width_pixels"] = _distance_between_points(
            left_ankle,
            right_ankle,
        )

    if left_shoulder and right_shoulder:
        measurements["shoulder_tilt_degrees"] = _angle_from_horizontal(
            left_shoulder,
            right_shoulder,
        )

    if left_hip and right_hip:
        measurements["hip_tilt_degrees"] = _angle_from_horizontal(left_hip, right_hip)
        measurements["hip_center"] = _midpoint(left_hip, right_hip)

    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_center = _midpoint(left_shoulder, right_shoulder)
        hip_center = _midpoint(left_hip, right_hip)
        measurements["torso_lean_degrees"] = _angle_from_vertical(
            hip_center,
            shoulder_center,
        )

    if left_wrist and right_wrist:
        measurements["hand_center"] = _midpoint(left_wrist, right_wrist)

    return measurements


def _record_frame_measurements(
    current_metrics,
    metric_history,
    head_positions,
    hip_center_positions,
    hand_center_positions,
):
    """Store one frame of metrics for the final summary."""
    for metric_name in metric_history:
        metric_value = current_metrics.get(metric_name)

        if metric_value is not None:
            metric_history[metric_name].append(metric_value)

    if current_metrics.get("head_position"):
        head_positions.append(current_metrics["head_position"])

    if current_metrics.get("hip_center"):
        hip_center_positions.append(current_metrics["hip_center"])

    if current_metrics.get("hand_center"):
        hand_center_positions.append(current_metrics["hand_center"])


def _maximum_absolute(values):
    """Return the largest absolute value in a list, preserving magnitude only."""
    if not values:
        return None

    return max(abs(value) for value in values)


def _build_analysis_summary(
    frame_number,
    detected_frame_count,
    missed_frame_count,
    metric_history,
    head_positions,
    hip_center_positions,
    hand_center_positions,
):
    """Create a readable, grouped summary from all tracked frames."""
    detection_rate = None
    if frame_number > 0:
        detection_rate = (detected_frame_count / frame_number) * 100

    return {
        "detection_quality": {
            "frames_processed": frame_number,
            "pose_detected_frames": detected_frame_count,
            "pose_missed_frames": missed_frame_count,
            "pose_detection_rate_percent": detection_rate,
        },
        "setup": {
            "average_stance_width_pixels": _average(
                metric_history["stance_width_pixels"]
            ),
        },
        "body_angles": {
            "average_shoulder_tilt_degrees": _average(
                metric_history["shoulder_tilt_degrees"]
            ),
            "maximum_shoulder_tilt_degrees": _maximum_absolute(
                metric_history["shoulder_tilt_degrees"]
            ),
            "average_hip_tilt_degrees": _average(metric_history["hip_tilt_degrees"]),
            "maximum_hip_tilt_degrees": _maximum_absolute(
                metric_history["hip_tilt_degrees"]
            ),
            "average_torso_lean_degrees": _average(
                metric_history["torso_lean_degrees"]
            ),
            "maximum_torso_lean_degrees": _maximum_absolute(
                metric_history["torso_lean_degrees"]
            ),
        },
        "movement": {
            "head_sway_pixels": _horizontal_range(head_positions),
            "hip_sway_pixels": _horizontal_range(hip_center_positions),
            "hand_path_pixels": _movement_range(hand_center_positions),
        },
        "notes": [
            "Front-view 2D video is useful for tilt and lateral sway.",
            "True shoulder turn, hip turn, X-factor, and weight shift need 3D data or another camera view.",
        ],
    }


def print_swing_analysis_results(analysis_results):
    """Print the swing analyzer output in a readable, grouped format."""
    print("\nSwing analysis results")
    print("=" * 30)

    section_titles = {
        "detection_quality": "Detection quality",
        "setup": "Setup",
        "body_angles": "Body angles",
        "movement": "Movement",
    }

    for section_name, section_title in section_titles.items():
        section_results = analysis_results.get(section_name, {})

        print(f"\n{section_title}")
        print("-" * len(section_title))

        for metric_name, metric_value in section_results.items():
            print(f"{_readable_metric_name(metric_name)}: {_format_summary_value(metric_value)}")

    notes = analysis_results.get("notes", [])
    if notes:
        print("\nNotes")
        print("-----")
        for note in notes:
            print(f"- {note}")


def _readable_metric_name(metric_name):
    """Convert internal metric names into terminal-friendly labels."""
    replacements = {
        "pixels": "px",
        "degrees": "deg",
        "percent": "%",
    }
    words = metric_name.split("_")

    for index, word in enumerate(words):
        words[index] = replacements.get(word, word)

    return " ".join(words).capitalize()


def _format_summary_value(metric_value):
    """Format final summary values consistently."""
    if metric_value is None:
        return "not enough data"

    if isinstance(metric_value, float):
        return f"{metric_value:.2f}"

    return str(metric_value)
