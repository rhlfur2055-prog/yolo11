# -*- coding: utf-8 -*-
"""
plate_gui_youtube_patch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plate_gui.py의 --youtube 옵션 연동 참조 (이미 적용됨).

[현재 적용 상태]
- plate_gui.py는 youtube_helper.download_youtube()를 사용합니다.
- argparse에 --youtube 인자가 있으며, main()에서 args.youtube 시
  youtube_helper.check_ytdlp() → download_youtube(args.youtube) 호출 후
  _open_video(video_path)로 재생합니다.

[수동 패치가 필요한 경우]
1. youtube_helper.py를 c:\\tools\\yolo26\\ 에 두기
2. plate_gui.py의 parser.add_argument 섹션에:
   parser.add_argument("--youtube", type=str, default=None,
                       help="YouTube URL → temp_youtube/에 다운로드 후 재생")
3. args 파싱 직후:
   if args.youtube:
       from youtube_helper import download_youtube, check_ytdlp
       if not check_ytdlp():
           print("[오류] pip install yt-dlp"); sys.exit(1)
       video_path = download_youtube(args.youtube)

[테스트]
  python plate_gui.py --youtube "https://youtube.com/watch?v=XFV56NOqr8A"
"""
