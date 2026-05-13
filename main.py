from video_loader import (
    get_video_info,
    open_video,
    print_video_info,
    release_video,
    validate_video_path,
)
from yolo_pose_analyzer import analyze_head_on_swing, print_swing_analysis_results


VIDEO_PATH = r"C:\Users\burke\OneDrive\Desktop\Funny Folder\Tom.mp4"


def run_head_on_swing_analyzer():
    """Run the first single-video head-on swing analyzer."""
    print("Open Golf Coach")
    print("================")
    print("Head-on golf swing analyzer using YOLO pose keypoints.\n")

    video_capture = None

    try:
        if not validate_video_path(VIDEO_PATH):
            print("Update VIDEO_PATH in main.py to point at a valid MP4 file.")
            return

        video_capture = open_video(VIDEO_PATH)
        if video_capture is None:
            return

        video_info = get_video_info(video_capture)
        print_video_info("Swing", video_info)

        analysis_results = analyze_head_on_swing(video_capture, VIDEO_PATH)
        print_swing_analysis_results(analysis_results)

    finally:
        release_video(video_capture)


if __name__ == "__main__":
    run_head_on_swing_analyzer()
