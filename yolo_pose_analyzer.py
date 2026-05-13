import math
import os
import site
import sys

import cv2
import numpy as np

NVIDIA_DLL_HANDLES = []


def _add_nvidia_dll_directories():
    """Let ONNX Runtime find CUDA/cuDNN DLLs installed by NVIDIA pip wheels."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    site_roots = site.getsitepackages() + [site.getusersitepackages()]
    nvidia_packages = ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn")

    for site_root in site_roots:
        for package_name in nvidia_packages:
            dll_directory = os.path.join(site_root, "nvidia", package_name, "bin")
            if os.path.isdir(dll_directory):
                os.environ["PATH"] = dll_directory + os.pathsep + os.environ.get("PATH", "")
                NVIDIA_DLL_HANDLES.append(os.add_dll_directory(dll_directory))


_add_nvidia_dll_directories()

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from rtmlib import Body
except ImportError:
    Body = None


WINDOW_NAME = "Open Golf Coach - Stabilized Swing Viewer"
MODEL_NAME = "RTMPose Body"
MIN_KEYPOINT_CONFIDENCE = 0.45
INFERENCE_EVERY_N_FRAMES = 1
INFERENCE_WIDTH = 960
RTMPOSE_MODE = "performance"
RTMPOSE_DET_INPUT_SIZE = (640, 640)
RTMPOSE_POSE_INPUT_SIZE = (288, 384)
RTMPOSE_PREFERRED_DEVICE = "cuda"
SMOOTHING_WINDOW = 9
PREDICTION_BLEND = 0.2
MAX_BODY_JUMP_FACTOR = 0.85
MAX_ARM_REACH_FACTOR = 2.8
MAX_LEG_REACH_FACTOR = 2.6
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
VIDEO_RECT = (30, 90, 1010, 760)
RIGHT_PANEL_LEFT = 1040
RIGHT_PANEL_RIGHT = 1570
COLOR_BACKGROUND = (14, 16, 22)
COLOR_PANEL = (26, 30, 39)
COLOR_PANEL_LIGHT = (34, 39, 50)
COLOR_TEXT = (238, 242, 248)
COLOR_MUTED = (155, 164, 178)
COLOR_ACCENT = (38, 214, 255)
COLOR_SUCCESS = (80, 235, 145)
COLOR_WARNING = (0, 190, 255)
COLOR_PURPLE = (210, 130, 255)
COLOR_ORANGE = (255, 180, 80)
COLOR_SHOULDER_BAR = (255, 255, 40)
COLOR_KEYPOINT = (70, 245, 230)
COLOR_LIMB_LINE = COLOR_ACCENT
COLOR_HIP_LINE = COLOR_PURPLE
COLOR_TORSO_LINE = COLOR_ORANGE

KEYPOINT_NAMES = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

POSE_SEGMENTS = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

ARM_KEYPOINTS = {
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}

LEG_KEYPOINTS = {
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


def analyze_head_on_swing(video_capture, video_path=None):
    """Analyze and replay one swing using accuracy-first RTMPose keypoints."""
    if Body is None:
        print("RTMLib is not installed. Run: python -m pip install rtmlib onnxruntime")
        return _empty_analysis_results()

    print("\nLoading RTMPose body model for accuracy-first analysis.")
    model = _load_rtmpose_model()

    print("Preprocessing the full video on GPU. Cache is disabled.")
    frames, raw_pose_sequence, detection_flags = _collect_video_pose_data(
        video_capture,
        model,
    )

    if not frames:
        print("No frames were read from the video.")
        return _empty_analysis_results()

    pose_sequence = _smooth_pose_sequence(raw_pose_sequence)
    metrics_sequence = [_calculate_frame_measurements(points) for points in pose_sequence]
    analysis_results = _build_analysis_summary(
        detection_flags,
        metrics_sequence,
        pose_sequence,
    )

    print("Replay ready.")
    print("Controls: p/space pause, a/d step while paused, scrub bar, q quit.")
    _replay_stabilized_swing(
        frames,
        pose_sequence,
        metrics_sequence,
        detection_flags,
    )

    return analysis_results


def _collect_video_pose_data(video_capture, model):
    """Read the video once and collect sampled RTMPose body points."""
    frames = []
    pose_sequence = []
    detection_flags = []
    frame_index = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        should_run_inference = frame_index % INFERENCE_EVERY_N_FRAMES == 0
        if should_run_inference:
            inference_frame, scale_x, scale_y = _resize_for_inference(frame)
            keypoints, scores = model(inference_frame)
            points = _extract_best_person_points(
                keypoints,
                scores,
                scale_x,
                scale_y,
                inference_frame.shape[1],
            )
        else:
            points = {}

        frames.append(frame)
        pose_sequence.append(points)
        detection_flags.append(bool(points) and should_run_inference)

        frame_index += 1
        if frame_index % 90 == 0:
            sampled_frames = frame_index // INFERENCE_EVERY_N_FRAMES
            print(f"Read {frame_index} frames, ran RTMPose on about {sampled_frames}...")

    return frames, pose_sequence, detection_flags


def _load_rtmpose_model():
    """Load RTMPose on CUDA when ONNX Runtime can see the GPU."""
    device = _select_rtmpose_device()
    print(f"Using RTMPose device: {device}")

    try:
        return Body(
            mode=RTMPOSE_MODE,
            det_input_size=RTMPOSE_DET_INPUT_SIZE,
            pose_input_size=RTMPOSE_POSE_INPUT_SIZE,
            backend="onnxruntime",
            device=device,
        )
    except Exception as error:
        if device == "cpu":
            raise

        print(f"Could not start RTMPose on CUDA: {error}")
        print("Falling back to CPU.")
        return Body(
            mode=RTMPOSE_MODE,
            det_input_size=RTMPOSE_DET_INPUT_SIZE,
            pose_input_size=RTMPOSE_POSE_INPUT_SIZE,
            backend="onnxruntime",
            device="cpu",
        )


def _select_rtmpose_device():
    """Prefer the RTX GPU when CUDA is available, otherwise use CPU."""
    if RTMPOSE_PREFERRED_DEVICE != "cuda" or ort is None:
        return "cpu"

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "cuda"

    return "cpu"


def _resize_for_inference(frame):
    """Resize a frame for faster pose inference and return scale factors."""
    frame_height, frame_width = frame.shape[:2]

    if frame_width <= INFERENCE_WIDTH:
        return frame, 1.0, 1.0

    scale = INFERENCE_WIDTH / frame_width
    inference_height = int(frame_height * scale)
    inference_frame = cv2.resize(frame, (INFERENCE_WIDTH, inference_height))

    return inference_frame, frame_width / INFERENCE_WIDTH, frame_height / inference_height


def _extract_best_person_points(
    keypoint_sets,
    confidence_sets,
    scale_x=1.0,
    scale_y=1.0,
    frame_width=None,
):
    """Extract the highest-confidence person's RTMPose body keypoints."""
    if keypoint_sets is None or confidence_sets is None or len(keypoint_sets) == 0:
        return {}

    best_index = _select_main_person_index(keypoint_sets, confidence_sets, frame_width)
    keypoints = keypoint_sets[best_index]
    confidences = confidence_sets[best_index]

    points = {}
    for name, index in KEYPOINT_NAMES.items():
        if confidences[index] < MIN_KEYPOINT_CONFIDENCE:
            points[name] = None
            continue

        x_value, y_value = keypoints[index]
        points[name] = [float(x_value * scale_x), float(y_value * scale_y)]

    return points


def _select_main_person_index(keypoint_sets, confidence_sets, frame_width=None):
    """Prefer the large, centered, high-confidence person over background detections."""
    person_scores = []
    image_center_x = (frame_width / 2) if frame_width else None

    for keypoints, confidences in zip(keypoint_sets, confidence_sets):
        valid_points = keypoints[confidences >= MIN_KEYPOINT_CONFIDENCE]
        if len(valid_points) < 4:
            person_scores.append(float("-inf"))
            continue

        min_x = float(np.min(valid_points[:, 0]))
        max_x = float(np.max(valid_points[:, 0]))
        min_y = float(np.min(valid_points[:, 1]))
        max_y = float(np.max(valid_points[:, 1]))
        box_width = max(max_x - min_x, 1.0)
        box_height = max(max_y - min_y, 1.0)
        box_area = box_width * box_height
        center_x = (min_x + max_x) / 2
        center_penalty = 0.0
        if image_center_x:
            center_penalty = abs(center_x - image_center_x) / max(image_center_x, 1.0)
        confidence_score = float(np.mean(confidences))

        person_scores.append(
            confidence_score
            + (0.18 * math.log1p(box_area))
            - (0.25 * center_penalty)
            + (0.08 * len(valid_points))
        )

    best_index = int(np.argmax(person_scores))
    if not math.isfinite(person_scores[best_index]):
        return int(np.argmax([confidences.mean() for confidences in confidence_sets]))

    return best_index


def _smooth_pose_sequence(raw_pose_sequence):
    """Stabilize the full pose sequence with prediction and body constraints."""
    frame_count = len(raw_pose_sequence)
    repaired_sequence = _repair_pose_sequence(raw_pose_sequence)
    constrained_sequence = [
        _apply_body_constraints(points) for points in repaired_sequence
    ]
    smoothed_sequence = [{name: None for name in KEYPOINT_NAMES} for _ in range(frame_count)]

    for name in KEYPOINT_NAMES:
        x_values = np.full(frame_count, np.nan)
        y_values = np.full(frame_count, np.nan)

        for frame_index, points in enumerate(constrained_sequence):
            point = points.get(name)
            if point is None:
                continue

            x_values[frame_index] = point[0]
            y_values[frame_index] = point[1]

        x_values = _interpolate_and_smooth_values(x_values)
        y_values = _interpolate_and_smooth_values(y_values)

        for frame_index in range(frame_count):
            if np.isnan(x_values[frame_index]) or np.isnan(y_values[frame_index]):
                continue

            smoothed_sequence[frame_index][name] = [
                float(x_values[frame_index]),
                float(y_values[frame_index]),
            ]

    return [_apply_body_constraints(points) for points in smoothed_sequence]


def _repair_pose_sequence(raw_pose_sequence):
    """Use predicted motion to reject sudden keypoint swaps and fill bad frames."""
    repaired_sequence = []
    previous_points = {}
    previous_velocity = {name: [0.0, 0.0] for name in KEYPOINT_NAMES}

    for raw_points in raw_pose_sequence:
        body_scale = _estimate_body_scale(raw_points) or _estimate_body_scale(previous_points) or 120
        repaired_points = {}

        for name in KEYPOINT_NAMES:
            raw_point = raw_points.get(name)
            previous_point = previous_points.get(name)
            velocity = previous_velocity.get(name, [0.0, 0.0])

            if previous_point is None:
                repaired_points[name] = raw_point
                continue

            predicted_point = [
                previous_point[0] + velocity[0],
                previous_point[1] + velocity[1],
            ]

            if raw_point is None:
                repaired_points[name] = previous_point
                continue

            jump_limit = _jump_limit_for_keypoint(name, body_scale)
            raw_jump = _distance_between_points(raw_point, predicted_point)

            if raw_jump > jump_limit:
                repaired_points[name] = previous_point
            else:
                repaired_points[name] = [
                    (PREDICTION_BLEND * predicted_point[index])
                    + ((1 - PREDICTION_BLEND) * raw_point[index])
                    for index in range(2)
                ]

        for name, repaired_point in repaired_points.items():
            previous_point = previous_points.get(name)

            if repaired_point is not None and previous_point is not None:
                previous_velocity[name] = [
                    repaired_point[0] - previous_point[0],
                    repaired_point[1] - previous_point[1],
                ]

        previous_points = repaired_points
        repaired_sequence.append(repaired_points)

    return repaired_sequence


def _jump_limit_for_keypoint(name, body_scale):
    """Allow faster hands, but reject large single-frame identity swaps."""
    if name in ARM_KEYPOINTS:
        return body_scale * MAX_BODY_JUMP_FACTOR * 1.35
    if name in LEG_KEYPOINTS:
        return body_scale * MAX_BODY_JUMP_FACTOR * 0.95
    return body_scale * MAX_BODY_JUMP_FACTOR * 0.75


def _apply_body_constraints(points):
    """Clamp outlier limbs back toward plausible body proportions."""
    constrained = {name: (point[:] if point else None) for name, point in points.items()}
    body_scale = _estimate_body_scale(constrained)

    if body_scale is None:
        return constrained

    left_shoulder = constrained.get("left_shoulder")
    right_shoulder = constrained.get("right_shoulder")
    left_hip = constrained.get("left_hip")
    right_hip = constrained.get("right_hip")

    if left_shoulder:
        _clamp_point_distance(constrained, "left_elbow", left_shoulder, body_scale * 1.35)
        _clamp_point_distance(constrained, "left_wrist", left_shoulder, body_scale * MAX_ARM_REACH_FACTOR)

    if right_shoulder:
        _clamp_point_distance(constrained, "right_elbow", right_shoulder, body_scale * 1.35)
        _clamp_point_distance(constrained, "right_wrist", right_shoulder, body_scale * MAX_ARM_REACH_FACTOR)

    if left_hip:
        _clamp_point_distance(constrained, "left_knee", left_hip, body_scale * 1.45)
        _clamp_point_distance(constrained, "left_ankle", left_hip, body_scale * MAX_LEG_REACH_FACTOR)

    if right_hip:
        _clamp_point_distance(constrained, "right_knee", right_hip, body_scale * 1.45)
        _clamp_point_distance(constrained, "right_ankle", right_hip, body_scale * MAX_LEG_REACH_FACTOR)

    return constrained


def _clamp_point_distance(points, point_name, anchor_point, max_distance):
    """Move an outlier point back onto a plausible radius from its anchor."""
    point = points.get(point_name)

    if point is None:
        return

    distance = _distance_between_points(point, anchor_point)
    if distance <= max_distance or distance == 0:
        return

    scale = max_distance / distance
    points[point_name] = [
        anchor_point[0] + ((point[0] - anchor_point[0]) * scale),
        anchor_point[1] + ((point[1] - anchor_point[1]) * scale),
    ]


def _estimate_body_scale(points):
    """Estimate body scale from shoulder width, hip width, or torso height."""
    if not points:
        return None

    candidate_scales = []
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")

    if left_shoulder and right_shoulder:
        candidate_scales.append(_distance_between_points(left_shoulder, right_shoulder))

    if left_hip and right_hip:
        candidate_scales.append(_distance_between_points(left_hip, right_hip) * 1.25)

    if left_shoulder and left_hip:
        candidate_scales.append(_distance_between_points(left_shoulder, left_hip) * 0.9)

    if right_shoulder and right_hip:
        candidate_scales.append(_distance_between_points(right_shoulder, right_hip) * 0.9)

    candidate_scales = [scale for scale in candidate_scales if scale > 20]
    if not candidate_scales:
        return None

    return float(np.median(candidate_scales))


def _interpolate_and_smooth_values(values):
    """Fill short detection gaps and apply a centered moving average."""
    valid_indexes = np.where(~np.isnan(values))[0]
    if len(valid_indexes) == 0:
        return values

    frame_indexes = np.arange(len(values))
    interpolated = np.interp(frame_indexes, valid_indexes, values[valid_indexes])

    window = min(SMOOTHING_WINDOW, len(interpolated))
    if window < 3:
        return interpolated

    if window % 2 == 0:
        window -= 1

    padding = window // 2
    padded = np.pad(interpolated, padding, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _calibrate_shoulders_with_user(frame, detected_points):
    """Let the user correct shoulder placement with two clicks."""
    left_shoulder = detected_points.get("left_shoulder")
    right_shoulder = detected_points.get("right_shoulder")

    if not (_is_finite_point(left_shoulder) and _is_finite_point(right_shoulder)):
        print("Shoulder calibration skipped because automatic shoulders were unavailable.")
        return {}

    print("\nShoulder calibration")
    print("Click the true LEFT shoulder, then the true RIGHT shoulder.")
    print("Press 's' in the calibration window to skip.")

    clicked_points = _collect_shoulder_clicks(frame, detected_points)
    if clicked_points is None:
        print("Shoulder calibration skipped.")
        return {}

    offsets = {
        "left_shoulder": [
            clicked_points["left_shoulder"][0] - left_shoulder[0],
            clicked_points["left_shoulder"][1] - left_shoulder[1],
        ],
        "right_shoulder": [
            clicked_points["right_shoulder"][0] - right_shoulder[0],
            clicked_points["right_shoulder"][1] - right_shoulder[1],
        ],
    }
    print("Shoulder calibration applied.")
    return offsets


def _collect_shoulder_clicks(frame, detected_points):
    """Collect two shoulder clicks on a scaled calibration frame."""
    window_name = "Shoulder Calibration"
    max_width = 1100
    frame_height, frame_width = frame.shape[:2]
    scale = min(max_width / frame_width, 1.0)
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)
    display_frame = cv2.resize(frame, (display_width, display_height))
    clicked_points = []
    labels = ["left_shoulder", "right_shoulder"]

    for name, point in detected_points.items():
        if _is_finite_point(point):
            display_point = (int(point[0] * scale), int(point[1] * scale))
            cv2.circle(display_frame, display_point, 4, COLOR_KEYPOINT, -1)

    detected_left = detected_points.get("left_shoulder")
    detected_right = detected_points.get("right_shoulder")
    if _is_finite_point(detected_left) and _is_finite_point(detected_right):
        cv2.line(
            display_frame,
            (int(detected_left[0] * scale), int(detected_left[1] * scale)),
            (int(detected_right[0] * scale), int(detected_right[1] * scale)),
            COLOR_SHOULDER_BAR,
            3,
        )

    def on_mouse(event, x_value, y_value, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked_points) >= 2:
            return

        clicked_points.append([x_value / scale, y_value / scale])
        cv2.circle(display_frame, (x_value, y_value), 8, COLOR_SHOULDER_BAR, -1)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while len(clicked_points) < 2:
        next_label = labels[len(clicked_points)].replace("_", " ")
        instruction_frame = display_frame.copy()
        _draw_text(
            instruction_frame,
            f"Click true {next_label} | s: skip",
            (20, 35),
            0.8,
            COLOR_SHOULDER_BAR,
            2,
        )
        cv2.imshow(window_name, instruction_frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("s"), ord("S"), ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return {
        "left_shoulder": clicked_points[0],
        "right_shoulder": clicked_points[1],
    }


def _apply_shoulder_offsets(pose_sequence, shoulder_offsets):
    """Apply manual shoulder calibration offsets to every frame."""
    if not shoulder_offsets:
        return pose_sequence

    calibrated_sequence = []
    for points in pose_sequence:
        calibrated_points = {
            name: (point[:] if point else None) for name, point in points.items()
        }

        for shoulder_name, offset in shoulder_offsets.items():
            shoulder_point = calibrated_points.get(shoulder_name)
            if shoulder_point is None:
                continue

            calibrated_points[shoulder_name] = [
                shoulder_point[0] + offset[0],
                shoulder_point[1] + offset[1],
            ]

        calibrated_sequence.append(calibrated_points)

    return calibrated_sequence


def _replay_stabilized_swing(
    frames,
    pose_sequence,
    metrics_sequence,
    detection_flags,
):
    """Replay the stabilized swing with pause, scrub, and multi-view model panels."""
    frame_count = len(frames)
    frame_index = 0
    paused = False
    shoulder_bar_multiplier = 1.0
    last_trackbar_position = 0

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(frame_count - 1, 1), _empty_callback)

    while True:
        trackbar_position = cv2.getTrackbarPos("Frame", WINDOW_NAME)
        if abs(trackbar_position - last_trackbar_position) > 1:
            frame_index = trackbar_position

        pose_points = pose_sequence[frame_index]
        metrics = metrics_sequence[frame_index]
        display_frame = _build_viewer_frame(
            frames[frame_index],
            pose_points,
            metrics,
            detection_flags[frame_index],
            paused,
            shoulder_bar_multiplier,
            frame_index + 1,
            frame_count,
        )
        cv2.imshow(WINDOW_NAME, display_frame)

        cv2.setTrackbarPos("Frame", WINDOW_NAME, frame_index)
        last_trackbar_position = frame_index

        key = cv2.waitKeyEx(40)
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P"), ord(" ")):
            paused = not paused
        if key in (ord("s"), ord("S")):
            shoulder_bar_multiplier = 2.0 if shoulder_bar_multiplier == 1.0 else 1.0
        if paused and key in (ord("a"), ord("A")):
            frame_index = max(0, frame_index - 1)
        elif paused and key in (ord("d"), ord("D")):
            frame_index = min(frame_count - 1, frame_index + 1)
        elif not paused:
            frame_index = (frame_index + 1) % frame_count

    cv2.destroyAllWindows()


def _build_viewer_frame(
    frame,
    points,
    metrics,
    pose_detected,
    paused,
    shoulder_bar_multiplier,
    frame_number,
    frame_count,
):
    """Compose the polished analysis UI into one OpenCV canvas."""
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), COLOR_BACKGROUND, dtype=np.uint8)
    _draw_header(canvas, paused, pose_detected)

    video_points = _draw_video_stage(canvas, frame, points)
    _draw_pose(canvas, video_points, shoulder_bar_multiplier)
    _draw_shoulder_check_card(
        canvas,
        points,
        (RIGHT_PANEL_LEFT, 90, RIGHT_PANEL_RIGHT, 300),
        shoulder_bar_multiplier,
    )
    _draw_metrics_strip(canvas, metrics)
    _draw_timeline(canvas, frame_number, frame_count, paused, shoulder_bar_multiplier)

    return canvas


def _draw_header(canvas, paused, pose_detected):
    """Draw the app title and status badges."""
    _draw_text(canvas, "OPEN GOLF COACH", (30, 38), 0.9, COLOR_TEXT, 2)
    _draw_text(canvas, "Stabilized 2D pose replay", (32, 66), 0.55, COLOR_MUTED, 1)

    detector_text = "RTMPOSE"
    status_text = "RAW DETECTED" if pose_detected else "INTERPOLATED"
    playback_text = "PAUSED" if paused else "PLAYING"

    _draw_badge(canvas, detector_text, (1040, 30), COLOR_ACCENT)
    _draw_badge(canvas, status_text, (1180, 30), COLOR_SUCCESS if pose_detected else COLOR_WARNING)
    _draw_badge(canvas, playback_text, (1370, 30), COLOR_PURPLE if paused else COLOR_SUCCESS)


def _draw_video_stage(canvas, frame, points):
    """Draw the video inside a clean stage and return transformed pose points."""
    left, top, right, bottom = VIDEO_RECT
    _draw_card(canvas, (left - 10, top - 10, right + 10, bottom + 10), COLOR_PANEL)
    _draw_text(canvas, "Original Video + 2D Pose Overlay", (left, top - 24), 0.55, COLOR_MUTED, 1)

    target_width = right - left
    target_height = bottom - top
    frame_height, frame_width = frame.shape[:2]
    scale = min(target_width / frame_width, target_height / frame_height)
    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)
    offset_x = left + ((target_width - resized_width) // 2)
    offset_y = top + ((target_height - resized_height) // 2)

    resized_frame = cv2.resize(frame, (resized_width, resized_height))
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized_frame

    transformed_points = {}
    for name, point in points.items():
        if not _is_finite_point(point) or not _point_inside_frame(point, frame_width, frame_height):
            transformed_points[name] = None
            continue

        transformed_points[name] = [
            offset_x + (point[0] * scale),
            offset_y + (point[1] * scale),
        ]

    cv2.rectangle(canvas, (offset_x, offset_y), (offset_x + resized_width, offset_y + resized_height), COLOR_PANEL_LIGHT, 1)
    return transformed_points


def _draw_model_card(
    canvas,
    points,
    rect,
    title,
    view_type,
    accent_color,
    shoulder_bar_multiplier,
):
    """Draw one model-view card in the analysis column."""
    _draw_card(canvas, rect, COLOR_PANEL)
    left, top, right, bottom = rect
    _draw_text(canvas, title, (left + 18, top + 32), 0.65, COLOR_TEXT, 2)

    if view_type == "top":
        subtitle = "Approximate path from one camera"
    else:
        subtitle = "Smoothed 2D reconstruction"
    _draw_text(canvas, subtitle, (left + 18, top + 58), 0.45, COLOR_MUTED, 1)

    model_points = _normalize_points_for_view(
        points,
        left + 35,
        top + 75,
        right - 35,
        bottom - 20,
        view_type,
    )
    _draw_pose_segments(canvas, model_points, accent_color, 3, shoulder_bar_multiplier)


def _draw_shoulder_check_card(canvas, points, rect, shoulder_bar_multiplier):
    """Draw a focused shoulder-line inspection card."""
    _draw_card(canvas, rect, COLOR_PANEL)
    left, top, right, bottom = rect
    _draw_text(canvas, "Shoulder Check", (left + 18, top + 32), 0.65, COLOR_TEXT, 2)
    _draw_text(
        canvas,
        f"Bright line: {shoulder_bar_multiplier:.0f}x shoulder width",
        (left + 18, top + 58),
        0.45,
        COLOR_MUTED,
        1,
    )

    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")

    if not (_is_finite_point(left_shoulder) and _is_finite_point(right_shoulder)):
        _draw_text(canvas, "Shoulders unavailable", (left + 18, top + 105), 0.55, COLOR_WARNING, 1)
        return

    tilt = _angle_from_horizontal(left_shoulder, right_shoulder)
    card_center = [(left + right) / 2, top + ((bottom - top) / 2) + 25]
    card_width = min((right - left) * 0.62 * shoulder_bar_multiplier, right - left - 80)
    half_width = card_width / 2

    vertical_offset = int(max(min((right_shoulder[1] - left_shoulder[1]) * 0.5, 35), -35))
    left_point = (int(card_center[0] - half_width), int(card_center[1] - vertical_offset))
    right_point = (int(card_center[0] + half_width), int(card_center[1] + vertical_offset))
    center_point = (int(card_center[0]), int(card_center[1]))

    cv2.line(canvas, left_point, right_point, COLOR_BACKGROUND, 14)
    cv2.line(canvas, left_point, right_point, COLOR_SHOULDER_BAR, 7)
    _draw_text(canvas, f"Tilt: {tilt:.1f} deg | Press s to toggle", (left + 18, bottom - 22), 0.55, COLOR_TEXT, 1)


def _draw_metrics_strip(canvas, metrics):
    """Draw compact metric cards across the lower left area."""
    cards = [
        ("Shoulder Tilt", _format_metric(metrics.get("shoulder_tilt_degrees"), " deg"), COLOR_SHOULDER_BAR),
        ("Hip Tilt", _format_metric(metrics.get("hip_tilt_degrees"), " deg"), COLOR_HIP_LINE),
        ("Torso Lean", _format_metric(metrics.get("torso_lean_degrees"), " deg"), COLOR_TORSO_LINE),
    ]

    start_x = 30
    y_top = 790
    card_width = 318
    card_height = 62

    for index, (label, value, color) in enumerate(cards):
        left = start_x + (index * (card_width + 15))
        rect = (left, y_top, left + card_width, y_top + card_height)
        _draw_card(canvas, rect, COLOR_PANEL)
        cv2.rectangle(canvas, (left, y_top), (left + 5, y_top + card_height), color, -1)
        _draw_text(canvas, label, (left + 18, y_top + 24), 0.48, COLOR_MUTED, 1)
        _draw_text(canvas, value, (left + 18, y_top + 52), 0.72, COLOR_TEXT, 2)


def _draw_timeline(canvas, frame_number, frame_count, paused, shoulder_bar_multiplier):
    """Draw replay controls and timeline progress."""
    y = 875
    left = 30
    right = 1570
    progress = 0 if frame_count <= 1 else (frame_number - 1) / (frame_count - 1)
    progress_x = int(left + ((right - left) * progress))

    cv2.line(canvas, (left, y), (right, y), COLOR_PANEL_LIGHT, 6)
    cv2.line(canvas, (left, y), (progress_x, y), COLOR_ACCENT, 6)
    cv2.circle(canvas, (progress_x, y), 9, COLOR_TEXT, -1)

    control_text = f"p/space pause | s shoulder {shoulder_bar_multiplier:.0f}x | a/d step when paused | scrub bar | q quit"
    frame_text = f"Frame {frame_number} / {frame_count}"
    if paused:
        control_text = f"Paused | a/d step | s shoulder {shoulder_bar_multiplier:.0f}x | p/space resume | q quit"

    _draw_text(canvas, control_text, (30, 846), 0.5, COLOR_MUTED, 1)
    _draw_text(canvas, frame_text, (1395, 846), 0.5, COLOR_MUTED, 1)


def _draw_card(canvas, rect, color):
    """Draw a rectangular card with a subtle border."""
    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), COLOR_PANEL_LIGHT, 1)


def _draw_badge(canvas, text, origin, color):
    """Draw a status badge in the header."""
    x, y = origin
    width = max(115, 14 * len(text))
    cv2.rectangle(canvas, (x, y - 20), (x + width, y + 10), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y - 20), (x + width, y + 10), color, 1)
    cv2.circle(canvas, (x + 15, y - 5), 5, color, -1)
    _draw_text(canvas, text, (x + 28, y), 0.45, COLOR_TEXT, 1)


def _draw_pose(frame, points, shoulder_bar_multiplier):
    """Draw the stabilized pose over the original video."""
    for start_name, end_name in POSE_SEGMENTS:
        start_point = points.get(start_name)
        end_point = points.get(end_name)

        if _is_finite_point(start_point) and _is_finite_point(end_point):
            cv2.line(
                frame,
                _as_int_tuple(start_point),
                _as_int_tuple(end_point),
                _pose_segment_color(start_name, end_name),
                4,
            )

    _draw_shoulder_bar(frame, points, shoulder_bar_multiplier)

    for point in points.values():
        if _is_finite_point(point):
            cv2.circle(frame, _as_int_tuple(point), 4, COLOR_KEYPOINT, -1)


def _draw_shoulder_bar(frame, points, shoulder_bar_multiplier):
    """Draw a clear stick line across the shoulders for visual checking."""
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")

    if not (_is_finite_point(left_shoulder) and _is_finite_point(right_shoulder)):
        return

    left_point, right_point, shoulder_center = _shoulder_bar_points(
        left_shoulder,
        right_shoulder,
        shoulder_bar_multiplier,
    )

    cv2.line(frame, left_point, right_point, COLOR_BACKGROUND, 10)
    cv2.line(frame, left_point, right_point, COLOR_SHOULDER_BAR, 5)


def _normalize_points_for_view(points, left, top, right, bottom, view_type):
    """Scale points into one of the model view panels."""
    visible_points = [point for point in points.values() if _is_finite_point(point)]
    if not visible_points:
        return {}

    min_x = min(point[0] for point in visible_points)
    max_x = max(point[0] for point in visible_points)
    min_y = min(point[1] for point in visible_points)
    max_y = max(point[1] for point in visible_points)
    source_width = max(max_x - min_x, 1)
    source_height = max(max_y - min_y, 1)

    target_width = max(right - left, 1)
    target_height = max(bottom - top, 1)
    scale = min(target_width / source_width, target_height / source_height)

    normalized = {}
    for name, point in points.items():
        if not _is_finite_point(point):
            normalized[name] = None
            continue

        x_normalized = (point[0] - min_x) * scale
        y_normalized = (point[1] - min_y) * scale

        if view_type == "back":
            x_normalized = target_width - x_normalized
        elif view_type == "top":
            # A single head-on camera cannot recover real depth. This compresses
            # vertical body position into a top-style path view for orientation.
            y_normalized = ((point[1] - min_y) / source_height) * target_height * 0.35

        normalized[name] = [
            left + ((target_width - source_width * scale) / 2) + x_normalized,
            top + ((target_height - source_height * scale) / 2) + y_normalized,
        ]

    return normalized


def _draw_pose_segments(frame, points, color, thickness, shoulder_bar_multiplier):
    """Draw a pose model into a side panel."""
    for start_name, end_name in POSE_SEGMENTS:
        start_point = points.get(start_name)
        end_point = points.get(end_name)

        if _is_finite_point(start_point) and _is_finite_point(end_point):
            cv2.line(
                frame,
                _as_int_tuple(start_point),
                _as_int_tuple(end_point),
                _pose_segment_color(start_name, end_name, color),
                thickness,
            )

    for point in points.values():
        if _is_finite_point(point):
            cv2.circle(frame, _as_int_tuple(point), 4, COLOR_KEYPOINT, -1)

    _draw_shoulder_bar(frame, points, shoulder_bar_multiplier)


def _pose_segment_color(start_name, end_name, default_color=COLOR_LIMB_LINE):
    """Use the same colors for body segments and their dashboard metrics."""
    segment = {start_name, end_name}

    if segment == {"left_shoulder", "right_shoulder"}:
        return COLOR_SHOULDER_BAR
    if segment == {"left_hip", "right_hip"}:
        return COLOR_HIP_LINE
    if "shoulder" in start_name and "hip" in end_name:
        return COLOR_TORSO_LINE
    if "hip" in start_name and "shoulder" in end_name:
        return COLOR_TORSO_LINE

    return default_color


def _shoulder_bar_points(left_shoulder, right_shoulder, shoulder_bar_multiplier):
    """Return shoulder guide endpoints expanded around the detected center."""
    center = _midpoint(left_shoulder, right_shoulder)
    half_vector = [
        (right_shoulder[0] - left_shoulder[0]) * 0.5 * shoulder_bar_multiplier,
        (right_shoulder[1] - left_shoulder[1]) * 0.5 * shoulder_bar_multiplier,
    ]
    left_point = [
        center[0] - half_vector[0],
        center[1] - half_vector[1],
    ]
    right_point = [
        center[0] + half_vector[0],
        center[1] + half_vector[1],
    ]

    return _as_int_tuple(left_point), _as_int_tuple(right_point), _as_int_tuple(center)


def _calculate_frame_measurements(points):
    """Calculate stable frame metrics from smoothed points."""
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")
    left_wrist = points.get("left_wrist")
    right_wrist = points.get("right_wrist")
    nose = points.get("nose")

    measurements = {
        "shoulder_tilt_degrees": None,
        "hip_tilt_degrees": None,
        "torso_lean_degrees": None,
        "head_position": nose,
        "hip_center": None,
        "hand_center": None,
    }

    if left_shoulder and right_shoulder:
        measurements["shoulder_tilt_degrees"] = _angle_from_horizontal(left_shoulder, right_shoulder)

    if left_hip and right_hip:
        measurements["hip_tilt_degrees"] = _angle_from_horizontal(left_hip, right_hip)
        measurements["hip_center"] = _midpoint(left_hip, right_hip)

    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_center = _midpoint(left_shoulder, right_shoulder)
        hip_center = _midpoint(left_hip, right_hip)
        measurements["torso_lean_degrees"] = _angle_from_vertical(hip_center, shoulder_center)

    if left_wrist and right_wrist:
        measurements["hand_center"] = _midpoint(left_wrist, right_wrist)

    return measurements


def _build_analysis_summary(detection_flags, metrics_sequence, pose_sequence):
    """Create grouped results from the smoothed replay data."""
    frame_count = len(detection_flags)
    detected_count = sum(1 for flag in detection_flags if flag)
    detection_rate = (detected_count / frame_count) * 100 if frame_count else None

    metric_history = {
        "shoulder_tilt_degrees": [],
        "hip_tilt_degrees": [],
        "torso_lean_degrees": [],
    }
    head_positions = []
    hip_center_positions = []
    hand_center_positions = []

    for metrics in metrics_sequence:
        for metric_name in metric_history:
            value = metrics.get(metric_name)
            if value is not None:
                metric_history[metric_name].append(value)

        if metrics.get("head_position"):
            head_positions.append(metrics["head_position"])
        if metrics.get("hip_center"):
            hip_center_positions.append(metrics["hip_center"])
        if metrics.get("hand_center"):
            hand_center_positions.append(metrics["hand_center"])

    return {
        "detection_quality": {
            "frames_processed": frame_count,
            "raw_pose_detected_frames": detected_count,
            "interpolated_frames": frame_count - detected_count,
            "raw_pose_detection_rate_percent": detection_rate,
        },
        "body_angles": {
            "average_shoulder_tilt_degrees": _average(metric_history["shoulder_tilt_degrees"]),
            "maximum_shoulder_tilt_degrees": _maximum_absolute(metric_history["shoulder_tilt_degrees"]),
            "average_hip_tilt_degrees": _average(metric_history["hip_tilt_degrees"]),
            "maximum_hip_tilt_degrees": _maximum_absolute(metric_history["hip_tilt_degrees"]),
            "average_torso_lean_degrees": _average(metric_history["torso_lean_degrees"]),
            "maximum_torso_lean_degrees": _maximum_absolute(metric_history["torso_lean_degrees"]),
        },
        "movement": {
            "head_sway_pixels": _horizontal_range(head_positions),
            "hip_sway_pixels": _horizontal_range(hip_center_positions),
            "hand_path_pixels": _movement_range(hand_center_positions),
        },
        "notes": [
            "The replay uses full-video interpolation and smoothing to reduce jitter.",
            "This viewer uses stabilized 2D RTMPose keypoints.",
        ],
    }


def print_swing_analysis_results(analysis_results):
    """Print the swing analyzer output in a readable, grouped format."""
    print("\nSwing analysis results")
    print("=" * 30)

    for section_name in ("detection_quality", "body_angles", "movement"):
        title = section_name.replace("_", " ").title()
        print(f"\n{title}")
        print("-" * len(title))

        for metric_name, metric_value in analysis_results.get(section_name, {}).items():
            print(f"{_readable_metric_name(metric_name)}: {_format_summary_value(metric_value)}")

    notes = analysis_results.get("notes", [])
    if notes:
        print("\nNotes")
        print("-----")
        for note in notes:
            print(f"- {note}")


def _empty_analysis_results():
    return {
        "detection_quality": {
            "frames_processed": 0,
            "raw_pose_detected_frames": 0,
            "interpolated_frames": 0,
            "raw_pose_detection_rate_percent": None,
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
        "notes": ["The analyzer could not run."],
    }


def _average(values):
    if not values:
        return None
    return sum(values) / len(values)


def _maximum_absolute(values):
    if not values:
        return None
    return max(abs(value) for value in values)


def _distance_between_points(point_a, point_b):
    x_difference = point_a[0] - point_b[0]
    y_difference = point_a[1] - point_b[1]
    return math.sqrt((x_difference * x_difference) + (y_difference * y_difference))


def _movement_range(points):
    if not points:
        return None

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return _distance_between_points([min(x_values), min(y_values)], [max(x_values), max(y_values)])


def _horizontal_range(points):
    if not points:
        return None

    x_values = [point[0] for point in points]
    return max(x_values) - min(x_values)


def _midpoint(point_a, point_b):
    return [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2]


def _angle_from_horizontal(point_a, point_b):
    angle = math.degrees(math.atan2(point_b[1] - point_a[1], point_b[0] - point_a[0]))

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def _angle_from_vertical(bottom_point, top_point):
    x_difference = top_point[0] - bottom_point[0]
    y_difference = bottom_point[1] - top_point[1]
    return math.degrees(math.atan2(x_difference, y_difference))


def _format_metric(value, suffix=""):
    if value is None:
        return "--"
    return f"{value:.1f}{suffix}"


def _format_summary_value(value):
    if value is None:
        return "not enough data"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _readable_metric_name(metric_name):
    replacements = {"pixels": "px", "degrees": "deg", "percent": "%"}
    words = [replacements.get(word, word) for word in metric_name.split("_")]
    return " ".join(words).capitalize()


def _is_finite_point(point):
    """Return True when a point is usable for drawing or measurement."""
    if point is None or len(point) < 2:
        return False

    return math.isfinite(point[0]) and math.isfinite(point[1])


def _point_inside_frame(point, frame_width, frame_height):
    """Reject points outside the source video to prevent runaway lines."""
    if not _is_finite_point(point):
        return False

    return 0 <= point[0] <= frame_width and 0 <= point[1] <= frame_height


def _draw_text(frame, text, position, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _as_int_tuple(point):
    return tuple(map(int, point))


def _empty_callback(position):
    return None
