#!/usr/bin/env python3
"""Generate thesis-ready qualitative figures from saved Meta-World GIFs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "thesis_figures"
VIDEO_DIR = ROOT / "logs" / "paper_eval_fix_20260413_232916" / "deltafix_video" / "video"

PANEL_SIZE = 480
LABEL_BAR = 42
PAD = 16
GAP = 12
BG = (255, 255, 255)
BORDER = (210, 216, 224)
TEXT = (35, 39, 47)
SUBTEXT = (88, 96, 110)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_LABEL = font(24, bold=True)
FONT_SMALL = font(19)


def gif_frame(path: Path, fraction: float) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        idx = max(0, min(im.n_frames - 1, round((im.n_frames - 1) * fraction)))
        im.seek(idx)
        return im.convert("RGB").resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont, fill=TEXT) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) // 2 - 1
    draw.text((x, y), text, font=fnt, fill=fill)


def paste_panel(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    y: int,
    title: str,
    subtitle: str | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + PANEL_SIZE - 1, y + LABEL_BAR + PANEL_SIZE - 1), outline=BORDER, width=1)
    draw.rectangle((x, y, x + PANEL_SIZE - 1, y + LABEL_BAR - 1), fill=(246, 248, 251), outline=BORDER, width=1)
    label = title if subtitle is None else f"{title}  |  {subtitle}"
    draw_centered(draw, (x, y, x + PANEL_SIZE, y + LABEL_BAR), label, FONT_LABEL if subtitle is None else FONT_SMALL)
    canvas.paste(image, (x, y + LABEL_BAR))


def make_task_overview() -> Path:
    tasks = [
        ("reach-v2", "reach-v2", 0.92),
        ("push-v2", "push-v2", 0.88),
        ("pick-place-v2", "pick-place-v2", 0.88),
        ("drawer-open-v2", "drawer-open-v2", 0.88),
        ("door-open-v2", "door-open-v2", 0.88),
        ("peg-insert-side-v2", "peg-insert-side-v2", 0.88),
    ]
    robot = "panda"
    width = PAD * 2 + PANEL_SIZE * 3 + GAP * 2
    height = PAD * 2 + (PANEL_SIZE + LABEL_BAR) * 2 + GAP
    canvas = Image.new("RGB", (width, height), BG)
    for i, (label, task, frac) in enumerate(tasks):
        gif = sorted(VIDEO_DIR.glob(f"{robot}_transformer_env0_{task}_success_1.0_reward_*_sample_0.gif"))[0]
        row, col = divmod(i, 3)
        x = PAD + col * (PANEL_SIZE + GAP)
        y = PAD + row * (PANEL_SIZE + LABEL_BAR + GAP)
        paste_panel(canvas, gif_frame(gif, frac), x, y, label)
    out = OUT_DIR / "fig_5_task_overview_panda.png"
    canvas.save(out, quality=95)
    return out


def make_success_sequences() -> Path:
    rows = [
        ("panda", "pick-place-v2", "panda / pick-place-v2", (0.03, 0.52, 0.91)),
        ("panda", "drawer-open-v2", "panda / drawer-open-v2", (0.03, 0.50, 0.88)),
        ("panda", "peg-insert-side-v2", "panda / peg-insert-side-v2", (0.03, 0.54, 0.90)),
    ]
    phases = ("Initial", "Middle", "Success")
    width = PAD * 2 + PANEL_SIZE * 3 + GAP * 2
    height = PAD * 2 + (PANEL_SIZE + LABEL_BAR) * len(rows) + GAP * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), BG)
    for r, (robot, task, label, fractions) in enumerate(rows):
        gif = sorted(VIDEO_DIR.glob(f"{robot}_transformer_env0_{task}_success_1.0_reward_*_sample_0.gif"))[0]
        for c, (phase, frac) in enumerate(zip(phases, fractions)):
            x = PAD + c * (PANEL_SIZE + GAP)
            y = PAD + r * (PANEL_SIZE + LABEL_BAR + GAP)
            paste_panel(canvas, gif_frame(gif, frac), x, y, label, phase)
    out = OUT_DIR / "fig_5_success_sequences_unseen_panda.png"
    canvas.save(out, quality=95)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(make_task_overview())
    print(make_success_sequences())


if __name__ == "__main__":
    main()
