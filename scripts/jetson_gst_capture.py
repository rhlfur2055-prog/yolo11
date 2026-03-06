#!/usr/bin/env python3
"""
Jetson GStreamer HW 디코딩 캡처 — CPU 점유율 최소화

nvv4l2decoder / nvvidconv 파이프라인으로 디코딩을 GPU에 맡겨
Headless 실행 시 사용할 수 있는 VideoCapture 예시.

사용법:
  from scripts.jetson_gst_capture import open_gst_capture
  cap = open_gst_capture("movie/hiway.mp4")
  while True:
      ret, frame = cap.read()
      ...
"""
import os
import sys

try:
    import cv2
except ImportError:
    cv2 = None


def _gst_pipeline_file(path: str, use_hw: bool = True) -> str:
    """파일 입력용 GStreamer 파이프라인 문자열.

    use_hw=True: Jetson에서 nvvidconv 등으로 HW 디코딩 (CPU 부하 감소).
    use_hw=False: decodebin만 사용 (호환성).
    """
    path = os.path.abspath(path)
    if use_hw and sys.platform.startswith("linux"):
        # Jetson: uridecodebin → nvvidconv → BGRx → appsink
        return (
            f"filesrc location={path} ! "
            "decodebin ! video/x-raw ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1"
        )
    return path  # 일반 파일 경로 → cv2.VideoCapture(path)


def open_gst_capture(source: str, use_hw: bool = True):
    """OpenCV VideoCapture. Jetson에서 source가 파일이면 GStreamer 파이프라인 시도."""
    if not cv2:
        raise RuntimeError("opencv-python required")
    if source.startswith(("rtsp://", "http://", "rtmp://")):
        return cv2.VideoCapture(source)
    if os.path.isfile(source) and use_hw and sys.platform.startswith("linux"):
        pipeline = _gst_pipeline_file(source, use_hw=True)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GStreamer capture test")
    p.add_argument("source", nargs="?", default="movie/hiway.mp4", help="Video file or RTSP URL")
    args = p.parse_args()
    cap = open_gst_capture(args.source)
    if not cap.isOpened():
        print("Failed to open source. Fallback to default VideoCapture.")
        cap = cv2.VideoCapture(args.source)
    print("Opened:", args.source)
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        print(f"  frame {i}: {frame.shape}")
    cap.release()
    print("Done.")
