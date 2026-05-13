import cv2

from video_loader import get_frame, get_video_info


ALIGNMENT_POINT_LABELS = [
    "ball_position",
    "left_foot",
    "right_foot",
    "golfer_center",
]


def ask_view_type():
    """Ask the user which camera view the video uses."""
    valid_view_types = {"face-on", "down-the-line"}

    while True:
        view_type = input("Choose camera view type (face-on/down-the-line): ").strip().lower()

        if view_type in valid_view_types:
            return view_type

        print("Please enter either 'face-on' or 'down-the-line'.")


def ask_frame_number(label):
    """Ask the user for a frame number, such as address or impact."""
    while True:
        raw_value = input(f"Enter the {label} frame number: ").strip()

        try:
            frame_number = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if frame_number < 0:
            print("Frame number cannot be negative.")
            continue

        return frame_number


def _mouse_click_callback(event, x, y, flags, click_data):
    """Store the next requested alignment point when the user clicks."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if click_data["current_index"] >= len(click_data["point_labels"]):
        return

    point_label = click_data["point_labels"][click_data["current_index"]]
    click_data["points"][point_label] = [x, y]
    click_data["current_index"] += 1

    cv2.circle(click_data["display_frame"], (x, y), 6, (0, 255, 0), -1)
    cv2.putText(
        click_data["display_frame"],
        point_label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def collect_click_points(frame, point_labels):
    """Collect named alignment points from mouse clicks on a video frame."""
    window_name = "Select alignment points"
    display_frame = frame.copy()
    click_data = {
        "point_labels": point_labels,
        "points": {},
        "current_index": 0,
        "display_frame": display_frame,
    }

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_click_callback, click_data)

    print("\nClick the requested points in the OpenCV window.")
    print("Press 'q' to cancel point selection.")

    while click_data["current_index"] < len(point_labels):
        current_label = point_labels[click_data["current_index"]]
        print(f"Click point: {current_label}")

        previous_index = click_data["current_index"]
        while click_data["current_index"] == previous_index:
            cv2.imshow(window_name, click_data["display_frame"])
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyWindow(window_name)
                print("Point selection cancelled.")
                return None

    cv2.imshow(window_name, click_data["display_frame"])
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)

    return click_data["points"]


def align_video(video_path, video_capture):
    """Collect manual alignment data for one golf swing video."""
    print(f"\nAlignment setup for: {video_path}")
    view_type = ask_view_type()
    address_frame = ask_frame_number("address")
    impact_frame = ask_frame_number("impact")

    video_info = get_video_info(video_capture)
    if address_frame >= video_info["frame_count"]:
        print("Address frame is outside the video frame range.")
        return None

    frame = get_frame(video_capture, address_frame)
    if frame is None:
        return None

    points = collect_click_points(frame, ALIGNMENT_POINT_LABELS)
    if points is None:
        return None

    return {
        "video_path": video_path,
        "view_type": view_type,
        "address_frame": address_frame,
        "impact_frame": impact_frame,
        "ball_position": points["ball_position"],
        "left_foot": points["left_foot"],
        "right_foot": points["right_foot"],
        "golfer_center": points["golfer_center"],
    }
