

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YouTube 영상 정보 조회
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_video_info(url: str) -> dict:
    """
    yt-dlp로 YouTube 영상 메타데이터 조회 (다운로드 없이)

    Args:
        url: YouTube 영상 URL

    Returns:
        {title, duration, width, height, fps, filesize_approx}
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp가 설치되어 있지 않습니다.\n"
            "설치: pip install yt-dlp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp 메타데이터 조회 시간 초과 (30초)")

    # Windows 한국어 환경 인코딩 호환: bytes → utf-8 디코딩 (오류 무시)
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    stderr_text = result.stderr.decode("utf-8", errors="replace")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 메타데이터 조회 실패:\n{stderr_text}")

    info = json.loads(stdout_text)
    return {
        "title": info.get("title", "알 수 없음"),
        "duration": info.get("duration", 0),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "fps": info.get("fps", 0),
        "filesize_approx": info.get("filesize_approx", 0),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YouTube 4K 영상 다운로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_youtube_4k(url: str, output_dir: str = "./temp_youtube") -> str:
    """
    YouTube 영상 4K 최고화질 다운로드

    [다운로드 해상도 전략]
    yt-dlp의 format selector를 활용한 Fallback 체인:

    1순위: bestvideo[height>=2160] + bestaudio → 4K (3840x2160)
    2순위: bestvideo[height>=1440] + bestaudio → QHD (2560x1440)
    3순위: bestvideo + bestaudio               → 가용 최고화질
    4순위: best                                → 단일 스트림 최고화질

    ※ 번호판 인식 정확도는 해상도에 비례하므로 4K를 최우선으로 시도.
    ※ 4K가 없는 영상에서도 자동으로 차선 해상도를 선택하여 다운로드.

    Args:
        url: YouTube 영상 URL
        output_dir: 다운로드 경로 (기본: ./temp_youtube)

    Returns:
        다운로드된 MP4 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title).80s.%(ext)s")

    # 4K 우선 → QHD → 최고화질 순서로 Fallback
    format_str = (
        "bestvideo[height>=2160]+bestaudio/best[height>=2160]/"
        "bestvideo[height>=1440]+bestaudio/best[height>=1440]/"
        "bestvideo+bestaudio/best"
    )

    cmd = [
        "yt-dlp",
        "-f", format_str,
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",            # 재생목록이 아닌 단일 영상만
        url,
    ]

    print(f"  [다운로드] yt-dlp 실행 중 (4K 우선)...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,  # 10분 타임아웃 (4K 영상은 용량이 큼)
        )
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp가 설치되어 있지 않습니다.\n"
            "설치: pip install yt-dlp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp 다운로드 시간 초과 (10분)")

    # Windows 한국어 환경 인코딩 호환
    stderr_text = result.stderr.decode("utf-8", errors="replace")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp 다운로드 실패:\n{stderr_text}")

    # output_dir에서 다운로드된 mp4 파일 탐색 (--print 대신 직접 탐색)
    filepath = None
    for f in os.listdir(output_dir):
        if f.endswith(".mp4"):
            candidate = os.path.join(output_dir, f)
            # 가장 최근 파일 선택
            if filepath is None or os.path.getmtime(candidate) > os.path.getmtime(filepath):
                filepath = candidate

    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(
            f"다운로드 완료되었으나 파일을 찾을 수 없습니다: {output_dir}"
        )

    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  [다운로드] 완료: {os.path.basename(filepath)} ({filesize_mb:.1f}MB)")

    return filepath


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def display_results(results: list[dict]) -> None:
    """인식 결과 테이블 출력"""
    print()
    print("=" * 80)
    print("  인식 결과 요약")
    print("=" * 80)

    if not results:
        print("  번호판을 찾지 못했습니다.")
        print("=" * 80)
        return

    valid_count = sum(1 for r in results if r.get("is_valid_plate"))
    print(f"  총 {len(results)}개 번호판 인식 (유효 패턴: {valid_count}개)\n")

    header = f"  {'#':<4} {'번호판 텍스트':<20} {'유효':<6} {'OCR':<7} {'패턴':<7} {'탐지':<7} {'방식':<14} {'프레임':<7}"
    print(header)
    print("  " + "-" * 72)

    for i, r in enumerate(results):
        valid_mark = "O" if r.get("is_valid_plate") else "-"
        method = r.get("detection_method", "?")
        print(
            f"  {i:<4} {r['text']:<20} "
            f"{valid_mark:<6} "
            f"{r['ocr_confidence']:.3f}  "
            f"{r.get('pattern_score', 0):.3f}  "
            f"{r['detection_confidence']:.3f}  "
            f"{method:<14} "
            f"{r['frame_idx']:>5}"
        )

    print("=" * 80)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 배너 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_banner(args: argparse.Namespace) -> None:
    """시작 배너 출력"""
    print()
    print("=" * 60)
    print("  YouTube 4K 번호판 인식 테스트 v2.0")
    print("  번호판 전용 YOLO + SAHI + EasyOCR + 한국 번호판 검증")
    print("=" * 60)
    print(f"  URL: {args.url}")
    print(f"  출력: {args.output}")
    print(f"  모델: {args.model or f'HuggingFace 자동 ({args.model_size})'}")
    print(f"  신뢰도: {args.confidence}")
    print(f"  SAHI: {'ON' if not args.no_sahi else 'OFF'}")
    print(f"  프레임 스킵: {args.frame_skip}")
    print(f"  영상 보존: {'ON' if args.keep_video else 'OFF (완료 후 삭제)'}")
    print("=" * 60)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    """
    YouTube 4K 번호판 인식 테스트 CLI

    사용법:
        python youtube_plate_test.py "https://www.youtube.com/watch?v=XXXX"
        python youtube_plate_test.py "URL" -o ./results --model yolo26s.pt
        python youtube_plate_test.py "URL" --keep-video --no-sahi
    """
    parser = argparse.ArgumentParser(
        description="YouTube 4K 영상 번호판 인식 테스트 (yt-dlp + YOLOv26 + EasyOCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python youtube_plate_test.py "https://www.youtube.com/watch?v=XXXX"
  python youtube_plate_test.py "URL" -o ./my_results --model yolo26s.pt
  python youtube_plate_test.py "URL" --keep-video --no-sahi --confidence 0.6
        """,
    )
    parser.add_argument(
        "url",
        help="YouTube 영상 URL",
    )
    parser.add_argument(
        "--output", "-o",
        default="./plate_results",
        help="결과 출력 디렉토리 (기본: ./plate_results)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="사용자 지정 YOLO 모델 .pt 경로 (미지정 시 HuggingFace 자동 다운로드)",
    )
    parser.add_argument(
        "--model-size",
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="번호판 모델 크기 (기본: n)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="최소 탐지 신뢰도 (기본: 0.3)",
    )
    parser.add_argument(
        "--no-sahi",
        action="store_true",
        help="SAHI 타일링 비활성화",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="SCANNING 상태 프레임 스킵 간격 (기본: 5)",
    )
    parser.add_argument(
        "--burst-frames",
        type=int,
        default=10,
        help="CAPTURING 버스트 캡처 프레임 수 (기본: 10)",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="다운로드 영상 유지 (기본: 처리 후 삭제)",
    )

    args = parser.parse_args()

    print_banner(args)
    start_time = time.time()

    # ── STEP 1: 영상 정보 조회 ──
    print("[1/3] YouTube 영상 정보 조회...")
    try:
        info = get_video_info(args.url)
        print(f"  제목: {info['title']}")
        print(f"  해상도: {info['width']}x{info['height']}")
        print(f"  길이: {info['duration']}초 ({info['duration'] / 60:.1f}분)")
        print(f"  FPS: {info['fps']}")

        if info["height"] >= 2160:
            print(f"  [확인] 4K(2160p) 해상도 사용 가능!")
        elif info["height"] >= 1440:
            print(f"  [참고] QHD(1440p) 해상도 (4K 미지원)")
        elif info["height"] >= 1080:
            print(f"  [참고] FHD(1080p) 해상도 (4K 미지원)")
        else:
            print(f"  [주의] {info['height']}p 저해상도 → 번호판 인식률 저하 가능")

    except Exception as e:
        print(f"  [경고] 정보 조회 실패: {e}")
        print(f"  다운로드를 계속 시도합니다...")

    # ── STEP 2: 4K 최고화질 다운로드 ──
    print(f"\n[2/3] 4K 최고화질 다운로드...")
    temp_dir = os.path.join(os.path.dirname(args.output), "temp_youtube")

    try:
        video_path = download_youtube_4k(args.url, temp_dir)
    except Exception as e:
        print(f"\n[오류] 다운로드 실패: {e}")
        sys.exit(1)

    # ── STEP 3: 번호판 인식 ──
    print(f"\n[3/3] 번호판 인식 시작...")

    # plate_recognition_4k.py에서 PlateRecognizer 임포트
    try:
        from plate_recognition_4k import PlateRecognizer
    except ImportError:
        # 동일 디렉토리에 있을 경우를 위한 경로 추가
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plate_recognition_4k import PlateRecognizer

    recognizer = PlateRecognizer(
        model_path=args.model,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        use_sahi=not args.no_sahi,
        frame_skip=args.frame_skip,
        burst_frames=args.burst_frames,
    )

    results = recognizer.process_video(video_path, args.output)

    # ── 결과 출력 ──
    display_results(results)

    # ── 임시 파일 정리 ──
    if not args.keep_video:
        print("\n임시 파일 정리...")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"  [삭제] {temp_dir}")
        except Exception:
            pass
    else:
        print(f"\n  [보존] 다운로드 영상: {video_path}")

    # ── 완료 ──
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  COMPLETE!")
    print("=" * 60)
    valid_count = sum(1 for r in results if r.get("is_valid_plate"))
    print(f"  인식 번호판: {len(results)}개 (유효 패턴: {valid_count}개)")
    print(f"  결과 저장: {args.output}")
    print(f"  총 소요: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
