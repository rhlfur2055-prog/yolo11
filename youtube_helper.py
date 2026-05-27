# -*- coding: utf-8 -*-
"""
youtube_helper.py — YouTube 영상 다운로드 헬퍼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plate_gui.py의 --youtube 옵션에서 호출.
yt-dlp로 temp_youtube/에 다운로드 후 로컬 파일 경로 반환.

사용법 (plate_gui.py 내부):
    from youtube_helper import download_youtube
    local_path = download_youtube(url)  # → "temp_youtube/영상제목.mp4"

단독 실행:
    python youtube_helper.py "https://youtube.com/watch?v=..."
"""

import os
import sys
import glob

# 다운로드 디렉토리 (프로젝트 루트/temp_youtube/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "temp_youtube")


def check_ytdlp() -> bool:
    """yt-dlp 설치 여부 확인."""
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def download_youtube(url: str, resolution: str = "best") -> str:
    """
    YouTube URL → 로컬 mp4 파일 다운로드.

    Args:
        url: YouTube 영상 URL
        resolution: 해상도 ("best", "1080", "720", "480")

    Returns:
        다운로드된 mp4 파일의 절대 경로

    Raises:
        ImportError: yt-dlp 미설치 시
        RuntimeError: 다운로드 실패 시
    """
    # yt-dlp 설치 확인
    if not check_ytdlp():
        print("[오류] yt-dlp가 설치되지 않았습니다.")
        print("       설치: pip install yt-dlp")
        raise ImportError("yt-dlp 미설치. pip install yt-dlp 로 설치하세요.")

    import yt_dlp

    # 다운로드 디렉토리 생성
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # 해상도별 포맷 선택
    if resolution == "best":
        format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    elif resolution in ("1080", "720", "480"):
        format_str = (
            f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={resolution}][ext=mp4]/best"
        )
    else:
        format_str = "best[ext=mp4]/best"

    # yt-dlp 옵션
    ydl_opts = {
        "format": format_str,
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [_progress_hook],
        # 이미 다운로드된 파일 스킵
        "download_archive": os.path.join(DOWNLOAD_DIR, ".archive.txt"),
    }

    print(f"[YouTube] 다운로드 시작: {url}")
    print(f"[YouTube] 저장 위치: {DOWNLOAD_DIR}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # 다운로드된 파일 경로 찾기
            if info:
                title = info.get("title", "video")
                # yt-dlp가 생성한 파일 찾기 (제목 기반)
                safe_title = yt_dlp.utils.sanitize_filename(title)
                expected = os.path.join(DOWNLOAD_DIR, f"{safe_title}.mp4")
                if os.path.isfile(expected):
                    print(f"[YouTube] 다운로드 완료: {expected}")
                    return os.path.abspath(expected)

                # 제목으로 못 찾으면 가장 최근 mp4 파일
                mp4_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.mp4"))
                if mp4_files:
                    latest = max(mp4_files, key=os.path.getmtime)
                    print(f"[YouTube] 다운로드 완료: {latest}")
                    return os.path.abspath(latest)

        raise RuntimeError("다운로드된 mp4 파일을 찾을 수 없습니다.")

    except Exception as e:
        print(f"[YouTube] 다운로드 실패: {e}")
        raise RuntimeError(f"YouTube 다운로드 실패: {e}") from e


def _progress_hook(d: dict) -> None:
    """yt-dlp 진행률 콜백."""
    status = d.get("status", "")
    if status == "downloading":
        pct = d.get("_percent_str", "?%").strip()
        speed = d.get("_speed_str", "?").strip()
        eta = d.get("_eta_str", "?").strip()
        print(f"\r[YouTube] {pct} | {speed} | ETA: {eta}", end="", flush=True)
    elif status == "finished":
        print(f"\n[YouTube] 다운로드 완료, 후처리 중...")


# 단독 실행 (테스트용)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python youtube_helper.py <YouTube URL>")
        print("예시:   python youtube_helper.py https://youtube.com/watch?v=XFV56NOqr8A")
        sys.exit(1)

    url = sys.argv[1]

    if not check_ytdlp():
        print("[오류] yt-dlp가 설치되지 않았습니다.")
        print("       설치: pip install yt-dlp")
        sys.exit(1)

    try:
        path = download_youtube(url)
        print(f"\n[완료] 파일 경로: {path}")
        print(f"[완료] plate_gui.py에서 열기:")
        print(f"       python plate_gui.py \"{path}\"")
    except Exception as e:
        print(f"\n[실패] {e}")
        sys.exit(1)
