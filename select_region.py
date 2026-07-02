"""Interactively pick a --region crop box for scanning.

Open a video, scrub to a frame that shows the text/logo you want to match,
draw a rectangle with the mouse, and the script prints the coordinates ready
to paste into ``main.py --region X Y W H``.

    python select_region.py video.mp4
    python select_region.py video.mp4 --time 12.5      # start at 12.5s
    python select_region.py video.mp4 --max-width 1600 # bigger preview

Controls (in the preview window):
    d / a        step forward / back one frame
    e / q        jump forward / back ~1 second
    w / s        jump forward / back ~10 seconds
    Enter / SPACE  draw the region on the current frame
    r            reset / clear the current selection
    Esc          quit
"""
import argparse
import sys
from pathlib import Path

import cv2


def _fit_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    """Scale factor (<=1.0) so a w*h frame fits inside max_w*max_h."""
    return min(1.0, max_w / w, max_h / h)


def _draw_hud(img, text: str):
    """Overlay a line of help text with a dark background strip."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(
        description="Interactively choose a --region crop box for scanning.")
    p.add_argument("input", help="video file to preview")
    p.add_argument("--time", type=float, default=0.0,
                   help="timestamp (seconds) of the first frame to show")
    p.add_argument("--max-width", type=int, default=1280,
                   help="max preview width in pixels (frame is scaled to fit)")
    p.add_argument("--max-height", type=int, default=720,
                   help="max preview height in pixels (frame is scaled to fit)")
    p.add_argument("--save-crop", metavar="PATH",
                   help="also write the cropped region to this image file")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Error: not found: {path}")

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = _fit_scale(full_w, full_h, args.max_width, args.max_height)

    print(f"Video: {full_w}x{full_h} @ {fps:.2f} fps  |  {total:,} frames")
    if scale < 1.0:
        print(f"Preview scaled to {scale:.3f}x to fit the screen "
              "(coordinates are reported at full resolution).")

    idx = max(0, min(total - 1, int(args.time * fps)))
    step_1s = max(1, int(round(fps)))
    step_10s = max(1, int(round(fps * 10)))

    win = "select-region  (d/a step, e/q ~1s, w/s ~10s, Enter=select, Esc=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def read_at(i: int):
        i = max(0, min(total - 1, i))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        return (i, frame) if ok else (i, None)

    region = None  # (x, y, w, h) at full resolution

    while True:
        idx, frame = read_at(idx)
        if frame is None:
            print(f"Warning: could not read frame {idx}", file=sys.stderr)
            idx = max(0, idx - 1)
            continue

        disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
        if region:
            x, y, w, h = region
            cv2.rectangle(disp,
                          (int(x * scale), int(y * scale)),
                          (int((x + w) * scale), int((y + h) * scale)),
                          (0, 255, 0), 2)
        t = idx / fps
        _draw_hud(disp, f"frame {idx}/{total - 1}   t={t:.2f}s   "
                        f"region={region if region else '(none)'}")
        cv2.imshow(win, disp)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:                       # Esc
            break
        elif key == ord("d"):
            idx += 1
        elif key == ord("a"):
            idx -= 1
        elif key == ord("e"):
            idx += step_1s
        elif key == ord("q"):
            idx -= step_1s
        elif key == ord("w"):
            idx += step_10s
        elif key == ord("s"):
            idx -= step_10s
        elif key == ord("r"):
            region = None
        elif key in (13, 32):               # Enter / Space -> select ROI
            sel = "select region, then Enter/Space to confirm (c to cancel)"
            roi = cv2.selectROI(sel, disp, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(sel)
            rx, ry, rw, rh = roi
            if rw and rh:
                # Map the display-space box back to full resolution.
                region = (int(rx / scale), int(ry / scale),
                          int(rw / scale), int(rh / scale))

    cap.release()
    cv2.destroyAllWindows()

    if not region:
        print("\nNo region selected.")
        return

    x, y, w, h = region
    # Clamp to frame bounds in case the box ran off the edge.
    x = max(0, min(x, full_w - 1))
    y = max(0, min(y, full_h - 1))
    w = max(1, min(w, full_w - x))
    h = max(1, min(h, full_h - y))

    print("\nSelected region (full resolution):")
    print(f"  X={x}  Y={y}  W={w}  H={h}")
    print("\nPaste into your scan command:")
    print(f"  --region {x} {y} {w} {h}")

    if args.save_crop:
        _, frame = read_at(idx)
        if frame is not None:
            cv2.imwrite(args.save_crop, frame[y:y + h, x:x + w])
            print(f"\nSaved crop preview: {args.save_crop}")


if __name__ == "__main__":
    main()
