import os

import cv2


def validate_video_path(video_path):
    """Check that the provided video path exists and points to an MP4 file."""
    if not video_path:
        print("No video path was provided.")
        return False

    if not os.path.isfile(video_path):
        print(f"Video file does not exist: {video_path}")
        return False

    if not video_path.lower().endswith(".mp4"):
        print("Video file must have a .mp4 extension.")
        return False

    return True


def open_video(video_path):
    """Open a video file with OpenCV and return the capture object."""
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    return video_capture


def get_video_info(video_capture):
    """Read basic video metadata from an opened cv2.VideoCapture object."""
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_seconds = 0
    if fps > 0:
        duration_seconds = frame_count / fps

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def get_frame(video_capture, frame_number):
    """Return a single frame by frame number, or None if it cannot be read."""
    if frame_number < 0:
        print("Frame number cannot be negative.")
        return None

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video_capture.read()

    if not success:
        print(f"Could not read frame number {frame_number}.")
        return None

    return frame


def print_video_info(video_name, info):
    """Print video metadata in a readable format."""
    print(f"\n{video_name} video information")
    print("-" * 30)
    print(f"FPS: {info['fps']:.2f}")
    print(f"Frame count: {info['frame_count']}")
    print(f"Width: {info['width']}")
    print(f"Height: {info['height']}")
    print(f"Duration: {info['duration_seconds']:.2f} seconds")


def release_video(video_capture):
    """Safely release an OpenCV video capture object."""
    if video_capture is not None:
        video_capture.release()
