"""Textual scan dashboard for vidgrep."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RichLog,
        Static,
    )
    from textual.worker import get_current_worker
except ModuleNotFoundError as exc:
    TEXTUAL_IMPORT_ERROR: Optional[ModuleNotFoundError] = exc
else:
    TEXTUAL_IMPORT_ERROR = None


COMPANION = r"""
 [o_o]
"""


@dataclass
class TuiDefaults:
    input_path: str = ""
    text: str = ""
    interval: str = ""
    region: str = ""
    no_gpu: bool = False


@dataclass
class ScanJob:
    input_path: Path
    pattern: str
    output: str
    region: Optional[tuple[int, int, int, int]]
    interval: Optional[float]
    skip_frames: int
    batch_size: int
    threshold: float
    language: str
    no_gpu: bool
    padding: float
    merge_gap: float
    min_duration: float
    concat: bool
    reencode: bool
    lossless: bool


@dataclass
class IntervalRecord:
    video_path: Path
    start: float
    end: float
    match_count: int
    best_confidence: float
    sample_text: str


@dataclass
class ScanState:
    matches: list[dict] = field(default_factory=list)
    intervals: list[IntervalRecord] = field(default_factory=list)


def _parse_region(value: str) -> Optional[tuple[int, int, int, int]]:
    value = value.strip()
    if not value:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("region must be four integers: X Y W H")
    return tuple(int(part) for part in parts)


def _parse_optional_float(value: str) -> Optional[float]:
    value = value.strip()
    return float(value) if value else None


def _matches_to_intervals(
    video_path: Path,
    matches: list[dict],
    *,
    merge_gap: float,
    min_duration: float,
) -> list[IntervalRecord]:
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda item: item["timestamp"])
    intervals: list[IntervalRecord] = []
    current: list[dict] = [sorted_matches[0]]

    for match in sorted_matches[1:]:
        if match["timestamp"] - current[-1]["timestamp"] <= merge_gap:
            current.append(match)
        else:
            _append_interval(intervals, video_path, current, min_duration)
            current = [match]

    _append_interval(intervals, video_path, current, min_duration)
    return intervals


def _append_interval(
    intervals: list[IntervalRecord],
    video_path: Path,
    matches: list[dict],
    min_duration: float,
) -> None:
    start = float(matches[0]["timestamp"])
    end = float(matches[-1]["timestamp"])
    if end - start < min_duration:
        return
    best = max(float(match["confidence"]) for match in matches)
    intervals.append(
        IntervalRecord(
            video_path=video_path,
            start=start,
            end=end,
            match_count=len(matches),
            best_confidence=best,
            sample_text=str(matches[0]["text"]),
        )
    )


if TEXTUAL_IMPORT_ERROR is None:

    class LogMessage(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    class ProgressMessage(Message):
        def __init__(self, sampled: int, total: int):
            super().__init__()
            self.sampled = sampled
            self.total = total

    class FileMessage(Message):
        def __init__(self, index: int, total: int, path: Path):
            super().__init__()
            self.index = index
            self.total = total
            self.path = path

    class MatchMessage(Message):
        def __init__(self, record: dict):
            super().__init__()
            self.record = record

    class ScanCompleteMessage(Message):
        def __init__(self, state: ScanState):
            super().__init__()
            self.state = state

    class ExtractCompleteMessage(Message):
        def __init__(self, paths: list[str]):
            super().__init__()
            self.paths = paths

    class FailureMessage(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    class VidgrepTui(App[None]):
        """Interactive scan dashboard for vidgrep."""

        CSS = """
        Screen {
            layout: vertical;
        }

        #body {
            height: 1fr;
        }

        #config {
            width: 38;
            min-width: 34;
            height: 100%;
            border: solid $primary;
            padding: 1;
        }

        #dashboard {
            width: 1fr;
            height: 100%;
        }

        #status {
            height: auto;
            min-height: 6;
            border: solid $accent;
            padding: 1;
        }

        #tables {
            height: 1fr;
        }

        #intervals {
            height: 1fr;
            border: solid $primary;
        }

        #log {
            height: 10;
            border: solid $primary;
        }

        Input {
            margin-bottom: 1;
        }

        Checkbox {
            margin-bottom: 1;
        }

        Button {
            width: 100%;
            margin-top: 1;
        }

        .half {
            width: 1fr;
        }

        #companion {
            color: $text-muted;
            width: 14;
            height: auto;
        }

        #status_text {
            width: 1fr;
            height: auto;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("ctrl+s", "start_scan", "Scan"),
            Binding("ctrl+e", "extract_all", "Extract All"),
        ]

        def __init__(self, defaults: TuiDefaults):
            super().__init__()
            self.defaults = defaults
            self.state = ScanState()
            self.last_job: Optional[ScanJob] = None
            self.scanning = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                with Vertical(id="config"):
                    yield Label("Input path")
                    yield Input(value=self.defaults.input_path, id="input_path")
                    yield Label("Text pattern")
                    yield Input(value=self.defaults.text, id="pattern")
                    yield Label("Output path/stem")
                    yield Input(id="output")
                    yield Label("Region X Y W H")
                    yield Input(value=self.defaults.region, id="region")
                    with Horizontal():
                        with Vertical(classes="half"):
                            yield Label("Interval seconds")
                            yield Input(value=self.defaults.interval, id="interval")
                        with Vertical(classes="half"):
                            yield Label("Skip frames")
                            yield Input(value="3", id="skip_frames")
                    with Horizontal():
                        with Vertical(classes="half"):
                            yield Label("Threshold")
                            yield Input(value="0.5", id="threshold")
                        with Vertical(classes="half"):
                            yield Label("Batch size")
                            yield Input(value="8", id="batch_size")
                    with Horizontal():
                        with Vertical(classes="half"):
                            yield Label("Padding")
                            yield Input(value="5", id="padding")
                        with Vertical(classes="half"):
                            yield Label("Merge gap")
                            yield Input(value="2", id="merge_gap")
                    yield Label("Minimum duration")
                    yield Input(value="0", id="min_duration")
                    yield Label("Language codes")
                    yield Input(value="en", id="language")
                    yield Checkbox("Disable GPU", value=self.defaults.no_gpu, id="no_gpu")
                    yield Checkbox("Concat clips", id="concat")
                    yield Checkbox("Re-encode NVENC", id="reencode")
                    yield Checkbox("Lossless libx264", id="lossless")
                    yield Button("Scan", variant="primary", id="scan")
                    yield Button("Extract selected interval", id="extract_selected", disabled=True)
                    yield Button("Extract all intervals", id="extract_all", disabled=True)
                with Vertical(id="dashboard"):
                    with Horizontal(id="status"):
                        yield Static(COMPANION, id="companion")
                        yield Static("Ready. Configure a text scan and press Scan.", id="status_text")
                    yield ProgressBar(total=100, id="progress")
                    with Vertical(id="tables"):
                        yield DataTable(id="intervals")
                    yield RichLog(id="log", wrap=True, highlight=False)
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#intervals", DataTable)
            table.add_columns("File", "Start", "End", "Hits", "Best", "Text")
            table.cursor_type = "row"
            table.zebra_stripes = True
            self.query_one("#progress", ProgressBar).update(progress=0, total=100)

        def action_start_scan(self) -> None:
            self._start_scan()

        def action_extract_all(self) -> None:
            self._extract_all()

        @on(Button.Pressed, "#scan")
        def on_scan_pressed(self) -> None:
            self._start_scan()

        @on(Button.Pressed, "#extract_all")
        def on_extract_all_pressed(self) -> None:
            self._extract_all()

        @on(Button.Pressed, "#extract_selected")
        def on_extract_selected_pressed(self) -> None:
            table = self.query_one("#intervals", DataTable)
            row = table.cursor_row
            if row is None or row >= len(self.state.intervals):
                self._log("No interval selected.")
                return
            self._extract([self.state.intervals[row]])

        def _start_scan(self) -> None:
            if self.scanning:
                self._log("A scan is already running.")
                return
            try:
                job = self._read_job()
            except ValueError as exc:
                self._set_status(f"Config error: {exc}")
                self._log(f"Config error: {exc}")
                return

            self.last_job = job
            self.state = ScanState()
            self.scanning = True
            self.query_one("#scan", Button).disabled = True
            self.query_one("#extract_selected", Button).disabled = True
            self.query_one("#extract_all", Button).disabled = True
            self.query_one("#intervals", DataTable).clear()
            self.query_one("#progress", ProgressBar).update(progress=0, total=100)
            self.query_one("#log", RichLog).clear()
            self._set_status("Scanning...")
            self._log("Scan started.")
            self.run_scan(job)

        def _read_job(self) -> ScanJob:
            input_text = self.query_one("#input_path", Input).value.strip()
            if not input_text:
                raise ValueError("input path is required")
            input_path = Path(input_text)
            if not input_path.exists():
                raise ValueError(f"not found: {input_path}")

            pattern = self.query_one("#pattern", Input).value.strip()
            if not pattern:
                raise ValueError("text pattern is required")

            interval = _parse_optional_float(self.query_one("#interval", Input).value)
            return ScanJob(
                input_path=input_path,
                pattern=pattern,
                output=self.query_one("#output", Input).value.strip(),
                region=_parse_region(self.query_one("#region", Input).value),
                interval=interval,
                skip_frames=max(1, int(self.query_one("#skip_frames", Input).value.strip() or "3")),
                batch_size=max(1, int(self.query_one("#batch_size", Input).value.strip() or "8")),
                threshold=float(self.query_one("#threshold", Input).value.strip() or "0.5"),
                language=self.query_one("#language", Input).value.strip() or "en",
                no_gpu=self.query_one("#no_gpu", Checkbox).value,
                padding=float(self.query_one("#padding", Input).value.strip() or "5"),
                merge_gap=float(self.query_one("#merge_gap", Input).value.strip() or "2"),
                min_duration=float(self.query_one("#min_duration", Input).value.strip() or "0"),
                concat=self.query_one("#concat", Checkbox).value,
                reencode=self.query_one("#reencode", Checkbox).value,
                lossless=self.query_one("#lossless", Checkbox).value,
            )

        @work(thread=True)
        def run_scan(self, job: ScanJob) -> None:
            from clipper import VideoClipper
            from detector import TextDetector
            from scan import find_video_files

            worker = get_current_worker()
            try:
                video_files = find_video_files(job.input_path)
                if not video_files:
                    self.post_message(FailureMessage(f"No valid video files found at '{job.input_path}'."))
                    return
                if job.output and len(video_files) > 1:
                    self.post_message(
                        FailureMessage("--output is only supported for extraction from one video in the TUI.")
                    )
                    return

                languages = [lang.strip() for lang in job.language.split(",") if lang.strip()]
                detector = TextDetector(
                    job.pattern,
                    gpu=not job.no_gpu,
                    threshold=job.threshold,
                    languages=languages,
                    log=lambda text: self.post_message(LogMessage(text)),
                )
                state = ScanState()

                for index, video_path in enumerate(video_files, 1):
                    if worker.is_cancelled:
                        return
                    self.post_message(FileMessage(index, len(video_files), video_path))
                    clipper = VideoClipper(
                        str(video_path),
                        log=lambda text: self.post_message(LogMessage(text)),
                    )
                    skip_frames = job.skip_frames
                    if job.interval:
                        skip_frames = max(1, round(job.interval * clipper.fps))
                        self.post_message(
                            LogMessage(
                                f"Sampling 1 frame every {job.interval:g}s "
                                f"(every {skip_frames} frames at {clipper.fps:.2f} fps)."
                            )
                        )

                    video_matches: list[dict] = []

                    def on_match(match: dict, _path=video_path) -> None:
                        record = {
                            "file": _path.name,
                            "path": str(_path.resolve()),
                            "timestamp": match["timestamp"],
                            "text": match["text"],
                            "confidence": match["confidence"],
                        }
                        video_matches.append(record)
                        state.matches.append(record)
                        self.post_message(MatchMessage(record))

                    clipper.scan_for_matches(
                        detector,
                        skip_frames=skip_frames,
                        region=job.region,
                        batch_size=job.batch_size,
                        collect_stats=True,
                        on_match=on_match,
                        show_progress=False,
                        on_progress=lambda sampled, total: self.post_message(
                            ProgressMessage(sampled, total)
                        ),
                    )
                    state.intervals.extend(
                        _matches_to_intervals(
                            video_path,
                            video_matches,
                            merge_gap=job.merge_gap,
                            min_duration=job.min_duration,
                        )
                    )

                self.post_message(ScanCompleteMessage(state))
            except Exception as exc:
                self.post_message(FailureMessage(str(exc)))

        @work(thread=True)
        def extract_intervals(self, job: ScanJob, intervals: list[IntervalRecord]) -> None:
            from clipper import VideoClipper

            try:
                saved: list[str] = []
                grouped: dict[Path, list[tuple[float, float]]] = {}
                for interval in intervals:
                    grouped.setdefault(interval.video_path, []).append((interval.start, interval.end))

                for video_path, spans in grouped.items():
                    output = job.output or None
                    clipper = VideoClipper(
                        str(video_path),
                        log=lambda text: self.post_message(LogMessage(text)),
                    )
                    saved.extend(
                        clipper.extract_clips(
                            spans,
                            padding=job.padding,
                            output=output,
                            reencode=job.reencode or job.lossless,
                            lossless=job.lossless,
                            concat=job.concat,
                        )
                    )
                self.post_message(ExtractCompleteMessage(saved))
            except SystemExit as exc:
                self.post_message(FailureMessage(str(exc)))
            except Exception as exc:
                self.post_message(FailureMessage(str(exc)))

        def _extract_all(self) -> None:
            if not self.state.intervals:
                self._log("No intervals to extract.")
                return
            self._extract(self.state.intervals)

        def _extract(self, intervals: list[IntervalRecord]) -> None:
            if self.last_job is None:
                self._log("No scan configuration available.")
                return
            self.query_one("#extract_selected", Button).disabled = True
            self.query_one("#extract_all", Button).disabled = True
            self._set_status(f"Extracting {len(intervals)} interval(s)...")
            self.extract_intervals(self.last_job, intervals)

        def on_log_message(self, message: LogMessage) -> None:
            self._log(message.text)

        def on_progress_message(self, message: ProgressMessage) -> None:
            total = max(message.total, 1)
            self.query_one("#progress", ProgressBar).update(
                total=total,
                progress=min(message.sampled, total),
            )

        def on_file_message(self, message: FileMessage) -> None:
            self._set_status(f"Scanning {message.index}/{message.total}: {message.path.name}")
            self._log(f"Scanning {message.path}")

        def on_match_message(self, message: MatchMessage) -> None:
            match = message.record
            self._log(
                f"{match['file']} @ {match['timestamp']:.3f}s: "
                f"{match['text']!r} ({match['confidence']:.2f})"
            )

        def on_scan_complete_message(self, message: ScanCompleteMessage) -> None:
            self.scanning = False
            self.state = message.state
            table = self.query_one("#intervals", DataTable)
            table.clear()
            for interval in self.state.intervals:
                table.add_row(
                    interval.video_path.name,
                    f"{interval.start:.2f}",
                    f"{interval.end:.2f}",
                    str(interval.match_count),
                    f"{interval.best_confidence:.2f}",
                    interval.sample_text,
                )
            self.query_one("#scan", Button).disabled = False
            has_intervals = bool(self.state.intervals)
            self.query_one("#extract_selected", Button).disabled = not has_intervals
            self.query_one("#extract_all", Button).disabled = not has_intervals
            self._set_status(
                f"Scan complete: {len(self.state.matches)} match(es), "
                f"{len(self.state.intervals)} interval(s)."
            )
            self._log("Scan complete.")

        def on_extract_complete_message(self, message: ExtractCompleteMessage) -> None:
            self.query_one("#extract_selected", Button).disabled = not self.state.intervals
            self.query_one("#extract_all", Button).disabled = not self.state.intervals
            self._set_status(f"Extracted {len(message.paths)} clip(s).")
            for path in message.paths:
                self._log(f"Saved: {path}")

        def on_failure_message(self, message: FailureMessage) -> None:
            self.scanning = False
            self.query_one("#scan", Button).disabled = False
            self.query_one("#extract_selected", Button).disabled = not self.state.intervals
            self.query_one("#extract_all", Button).disabled = not self.state.intervals
            self._set_status(f"Error: {message.text}")
            self._log(f"Error: {message.text}")

        def _set_status(self, text: str) -> None:
            self.query_one("#status_text", Static).update(text)

        def _log(self, text: str) -> None:
            self.query_one("#log", RichLog).write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidgrep-tui",
        description="Open the interactive vidgrep scan dashboard.",
    )
    parser.add_argument("input", nargs="?", help="Optional input video file or directory")
    parser.add_argument("--text", "-t", default="", help="Initial text pattern")
    parser.add_argument("--interval", "-i", type=float, help="Initial interval seconds")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-gpu", action="store_true", help="Start with GPU disabled")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if TEXTUAL_IMPORT_ERROR is not None:
        sys.exit(
            "Textual is required for vidgrep-tui. Install it with:\n"
            "  python -m pip install -e \".[tui]\""
        )

    defaults = TuiDefaults(
        input_path=args.input or "",
        text=args.text,
        interval="" if args.interval is None else f"{args.interval:g}",
        region="" if args.region is None else " ".join(str(part) for part in args.region),
        no_gpu=args.no_gpu,
    )
    VidgrepTui(defaults).run()


if __name__ == "__main__":
    main()
