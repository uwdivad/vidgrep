import sys
from pathlib import Path

from args import build_parser


def main():
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: file not found: {input_path}")

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

    clipper = VideoClipper(str(input_path))

    print(f"\nScanning '{input_path.name}' …")
    region = tuple(args.region) if args.region else None
    intervals = clipper.find_intervals(
        detector,
        skip_frames=args.skip_frames,
        region=region,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    if not intervals:
        sys.exit("\nNo matches found.")

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

    print()
    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
