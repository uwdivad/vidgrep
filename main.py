import sys
from pathlib import Path

from args import build_parser


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        from args import build_scan_parser
        from scan import run_scan
        run_scan(build_scan_parser().parse_args(sys.argv[2:]))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "inventory":
        from args import build_inventory_parser
        from inventory import run_inventory
        run_inventory(build_inventory_parser().parse_args(sys.argv[2:]))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        from args import build_worker_parser
        from worker import run_worker
        run_worker(build_worker_parser().parse_args(sys.argv[2:]))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        from args import build_agent_parser
        from agent import run_agent
        run_agent(build_agent_parser().parse_args(sys.argv[2:]))
        return

    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: not found: {input_path}")

    from scan import find_video_files

    try:
        video_files = find_video_files(input_path)
    except FileNotFoundError as exc:
        sys.exit(f"Error: {exc}")

    if not video_files:
        sys.exit(f"Error: no valid video files found at '{input_path}'")

    if input_path.is_dir() and args.output:
        sys.exit(
            "Error: --output is not allowed when input is a directory "
            "(clips are named per-file next to each source)."
        )

    from detector import TextDetector, TemplateDetector
    from clipper import VideoClipper

    use_gpu = not args.no_gpu
    languages = [lang.strip() for lang in args.lang.split(",")]

    if args.text:
        detector = TextDetector(
            args.text,
            gpu=use_gpu,
            threshold=args.threshold,
            languages=languages,
        )
    else:
        detector = TemplateDetector(
            args.template,
            gpu=use_gpu,
            threshold=args.threshold,
        )

    region = tuple(args.region) if args.region else None
    multiple = len(video_files) > 1
    if multiple:
        print(f"Found {len(video_files)} video file(s) to process.")

    saved_paths: list[str] = []
    for video_path in video_files:
        print(f"\nScanning '{video_path.name}' …")
        try:
            clipper = VideoClipper(str(video_path))
        except ValueError as exc:
            print(f"Warning: skipping '{video_path}': {exc}", file=sys.stderr)
            continue

        if args.interval:
            skip_frames = max(1, round(args.interval * clipper.fps))
            print(
                f"Sampling 1 frame every {args.interval:g}s "
                f"(every {skip_frames} frames at {clipper.fps:.2f} fps)."
            )
        else:
            skip_frames = args.skip_frames

        intervals = clipper.find_intervals(
            detector,
            skip_frames=skip_frames,
            region=region,
            merge_gap=args.merge_gap,
            min_duration=args.min_duration,
            batch_size=args.batch_size,
            collect_stats=args.stats,
        )

        if not intervals:
            print("No matches found.")
            continue

        print(f"\nFound {len(intervals)} interval(s):")
        for i, (s, e) in enumerate(intervals, 1):
            print(f"  [{i}]  {s:.2f}s – {e:.2f}s  (match duration: {e - s:.2f}s)")

        output_paths = clipper.extract_clips(
            intervals,
            padding=args.padding,
            output=args.output,
            reencode=args.reencode or args.lossless,
            lossless=args.lossless,
            concat=args.concat,
        )
        saved_paths.extend(output_paths)

    print()
    if not saved_paths:
        sys.exit("No clips written — no matches in any input.")
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
