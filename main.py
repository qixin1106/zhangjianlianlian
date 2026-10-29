from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import Quartz
from PIL import Image
from pynput import keyboard

ROWS = 9
COLS = 7
UNKNOWN = -1
SPECIAL = 5
REFERENCE_WIDTH = 1262
REFERENCE_HEIGHT = 1880
REFERENCE_XS = np.array([242, 371, 500, 630, 759, 888, 1017], dtype=np.float64)
REFERENCE_YS = np.array(
    [505, 634, 764, 893, 1022, 1152, 1281, 1410, 1540], dtype=np.float64
)
PIECE_NAMES = ("pumpkin", "tree", "castle", "bat", "ghost", "special")
DISPLAY_NAMES = {
    "pumpkin": "南瓜",
    "tree": "树桩",
    "castle": "城堡",
    "bat": "蝙蝠",
    "ghost": "幽灵",
    "special": "鬼脸",
    "unknown": "未知",
}
ROOT = Path(__file__).resolve().parent

Cell = tuple[int, int]


@dataclass(frozen=True)
class BoardGeometry:
    width: int
    height: int

    @classmethod
    def from_image(cls, image: np.ndarray) -> BoardGeometry:
        height, width = image.shape[:2]
        return cls(width=width, height=height)

    @property
    def scale_x(self) -> float:
        return self.width / REFERENCE_WIDTH

    @property
    def scale_y(self) -> float:
        return self.height / REFERENCE_HEIGHT

    @property
    def cell_size(self) -> int:
        return max(32, round(144 * self.scale_x))

    @property
    def template_size(self) -> int:
        return max(24, round(112 * self.scale_x))

    def center(self, row: int, col: int) -> tuple[int, int]:
        return (
            round(REFERENCE_XS[col] * self.scale_x),
            round(REFERENCE_YS[row] * self.scale_y),
        )

    def crop(self, image: np.ndarray, row: int, col: int) -> np.ndarray:
        cx, cy = self.center(row, col)
        half = self.cell_size // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(self.width, cx + half), min(self.height, cy + half)
        cell = image[y1:y2, x1:x2]
        if cell.shape[:2] != (half * 2, half * 2):
            raise ValueError(f"格子超出图像范围: row={row}, col={col}, shape={cell.shape}")
        return cell


@dataclass(frozen=True)
class BoardReading:
    labels: np.ndarray
    confidence: np.ndarray
    margins: np.ndarray
    stable: np.ndarray


class TemplateClassifier:
    cell_size = 36
    source_cell_size = 144

    def __init__(
        self,
        template_dir: Path,
        geometry: BoardGeometry,
        threshold: float = 0.72,
        min_margin: float = 0.025,
    ) -> None:
        self.threshold = threshold
        self.min_margin = min_margin
        self.templates: list[np.ndarray] = []
        for name in PIECE_NAMES:
            path = template_dir / f"{name}.png"
            template = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if template is None:
                raise FileNotFoundError(f"缺少棋子模板: {path}")
            crop_size = 64 if name == "special" else 80
            x = (template.shape[1] - crop_size) // 2
            y = (template.shape[0] - crop_size) // 2
            template = template[y : y + crop_size, x : x + crop_size]
            size = max(8, round(crop_size * self.cell_size / self.source_cell_size))
            self.templates.append(
                cv2.resize(template, (size, size), interpolation=cv2.INTER_AREA)
            )

    def classify_cell(self, cell: np.ndarray) -> tuple[int, float, float]:
        cell = cv2.resize(
            cell,
            (self.cell_size, self.cell_size),
            interpolation=cv2.INTER_AREA,
        )
        scores: list[float] = []
        for template in self.templates:
            result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
            _, best, _, _ = cv2.minMaxLoc(result)
            scores.append(float(best))
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_score = scores[best_index]
        margin = best_score - scores[int(order[1])]
        if best_score < self.threshold or margin < self.min_margin:
            return UNKNOWN, best_score, margin
        return best_index, best_score, margin

    def recognize(self, image: np.ndarray) -> BoardReading:
        geometry = BoardGeometry.from_image(image)
        labels = np.full((ROWS, COLS), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((ROWS, COLS), dtype=np.float32)
        margins = np.zeros((ROWS, COLS), dtype=np.float32)
        for row in range(ROWS):
            for col in range(COLS):
                label, score, margin = self.classify_cell(geometry.crop(image, row, col))
                labels[row, col] = label
                confidence[row, col] = score
                margins[row, col] = margin
        return BoardReading(
            labels=labels,
            confidence=confidence,
            margins=margins,
            stable=labels >= 0,
        )


@dataclass(frozen=True)
class MatchGroup:
    label: int
    direction: str
    cells: frozenset[Cell]

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class Move:
    first: Cell
    second: Cell
    immediate_count: int
    guaranteed_count: int
    longest_run: int
    special: bool = False

    @property
    def rank(self) -> tuple[int, int, int, int, int, int, int]:
        lowest = max(self.first[0], self.second[0])
        return (
            int(self.special),
            self.guaranteed_count,
            self.immediate_count,
            self.longest_run,
            lowest,
            -self.first[0],
            -self.first[1],
        )


def find_matches(board: np.ndarray) -> list[MatchGroup]:
    groups: list[MatchGroup] = []
    for row in range(ROWS):
        start = 0
        while start < COLS:
            label = int(board[row, start])
            end = start + 1
            while (
                end < COLS
                and 0 <= label != SPECIAL
                and int(board[row, end]) == label
            ):
                end += 1
            if 0 <= label != SPECIAL and end - start >= 3:
                groups.append(
                    MatchGroup(
                        label=label,
                        direction="horizontal",
                        cells=frozenset((row, col) for col in range(start, end)),
                    )
                )
            start = end
    for col in range(COLS):
        start = 0
        while start < ROWS:
            label = int(board[start, col])
            end = start + 1
            while (
                end < ROWS
                and 0 <= label != SPECIAL
                and int(board[end, col]) == label
            ):
                end += 1
            if 0 <= label != SPECIAL and end - start >= 3:
                groups.append(
                    MatchGroup(
                        label=label,
                        direction="vertical",
                        cells=frozenset((row, col) for row in range(start, end)),
                    )
                )
            start = end
    return groups


def collapse(board: np.ndarray, cleared: set[Cell]) -> np.ndarray:
    result = np.full_like(board, UNKNOWN)
    for col in range(COLS):
        remaining = [
            int(board[row, col]) for row in range(ROWS) if (row, col) not in cleared
        ]
        offset = ROWS - len(remaining)
        for index, label in enumerate(remaining):
            result[offset + index, col] = label
    return result


def evaluate_move(
    labels: np.ndarray,
    stable: np.ndarray,
    first: Cell,
    second: Cell,
) -> Move | None:
    if labels[first] < 0 or labels[second] < 0:
        return None
    if not stable[first] or not stable[second]:
        return None
    if labels[first] == SPECIAL or labels[second] == SPECIAL:
        return Move(
            first=first,
            second=second,
            immediate_count=0,
            guaranteed_count=0,
            longest_run=0,
            special=True,
        )
    if labels[first] == labels[second]:
        return None
    work = np.where(stable, labels, UNKNOWN).astype(np.int8)
    work[first], work[second] = work[second], work[first]
    groups = find_matches(work)
    new_groups = [group for group in groups if {first, second} & group.cells]
    if not new_groups:
        return None
    immediate = set().union(*(group.cells for group in new_groups))
    longest = max(group.length for group in new_groups)
    guaranteed = set(immediate)
    cleared = set(immediate)
    for _ in range(ROWS):
        work = collapse(work, cleared)
        next_groups = find_matches(work)
        if not next_groups:
            break
        cleared = set().union(*(group.cells for group in next_groups))
        guaranteed.update(cleared)
    return Move(
        first=first,
        second=second,
        immediate_count=len(immediate),
        guaranteed_count=len(guaranteed),
        longest_run=longest,
    )


def find_moves(labels: np.ndarray, stable: np.ndarray | None = None) -> list[Move]:
    if labels.shape != (ROWS, COLS):
        raise ValueError(f"棋盘尺寸错误: {labels.shape}")
    if stable is None:
        stable = labels >= 0
    moves: list[Move] = []
    for row in range(ROWS):
        for col in range(COLS):
            first = (row, col)
            neighbors = []
            if col + 1 < COLS:
                neighbors.append((row, col + 1))
            if row + 1 < ROWS:
                neighbors.append((row + 1, col))
            for second in neighbors:
                move = evaluate_move(labels, stable, first, second)
                if move is not None:
                    moves.append(move)
    return sorted(moves, key=lambda move: move.rank, reverse=True)


def format_board(labels: np.ndarray) -> str:
    names = [DISPLAY_NAMES[name] for name in PIECE_NAMES]
    lines = []
    for row in range(ROWS):
        lines.append(
            " ".join(
                names[int(label)] if label >= 0 else DISPLAY_NAMES["unknown"]
                for label in labels[row]
            )
        )
    return "\n".join(lines)


def format_move(move: Move) -> str:
    first = (move.first[0] + 1, move.first[1] + 1)
    second = (move.second[0] + 1, move.second[1] + 1)
    if move.special:
        return f"{first} -> {second}，绿色鬼脸优先交换"
    return (
        f"{first} -> {second}，立即消除 {move.immediate_count}，"
        f"确定总消除 {move.guaranteed_count}，最长匹配 {move.longest_run}"
    )


def load_classifier(image: np.ndarray, threshold: float) -> TemplateClassifier:
    return TemplateClassifier(
        ROOT / "images" / "templates",
        BoardGeometry.from_image(image),
        threshold=threshold,
    )


@dataclass(frozen=True)
class WindowFrame:
    image: np.ndarray
    x: float
    y: float
    logical_width: float
    logical_height: float

    @property
    def scale(self) -> float:
        return self.image.shape[1] / self.logical_width

    def screen_point(self, physical_x: float, physical_y: float) -> tuple[float, float]:
        return self.x + physical_x / self.scale, self.y + physical_y / self.scale


def find_game_window() -> dict:
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    candidates = []
    for window in windows:
        if window.get(Quartz.kCGWindowOwnerName) != "杖剑传说":
            continue
        bounds = window.get(Quartz.kCGWindowBounds, {})
        if bounds.get("Width", 0) >= 300 and bounds.get("Height", 0) >= 500:
            candidates.append(window)
    if not candidates:
        raise RuntimeError("没有找到可见的杖剑传说窗口")
    return max(
        candidates,
        key=lambda window: window[Quartz.kCGWindowBounds]["Width"]
        * window[Quartz.kCGWindowBounds]["Height"],
    )


def capture_game_window() -> WindowFrame:
    window = find_game_window()
    bounds = window[Quartz.kCGWindowBounds]
    x, y = float(bounds["X"]), float(bounds["Y"])
    width, height = float(bounds["Width"]), float(bounds["Height"])
    image_ref = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window[Quartz.kCGWindowNumber],
        Quartz.kCGWindowImageBoundsIgnoreFraming,
    )
    if image_ref is None:
        screenshot = pyautogui.screenshot(
            region=(round(x), round(y), round(width), round(height))
        )
        rgb = np.asarray(screenshot)
    else:
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image_ref))
        pil_image = Image.frombytes(
            "RGB",
            (
                Quartz.CGImageGetWidth(image_ref),
                Quartz.CGImageGetHeight(image_ref),
            ),
            bytes(data),
            "raw",
            "BGRX",
            Quartz.CGImageGetBytesPerRow(image_ref),
            1,
        )
        rgb = np.asarray(pil_image)
    return WindowFrame(
        image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        x=x,
        y=y,
        logical_width=width,
        logical_height=height,
    )


STATE_REGIONS = {
    "start": (390, 1430, 480, 210),
    "success": (330, 300, 600, 230),
}
STATE_TEMPLATE_CACHE: dict[str, np.ndarray] = {}


def state_score(image: np.ndarray, name: str) -> float:
    if name not in STATE_TEMPLATE_CACHE:
        template = cv2.imread(str(ROOT / "images" / "templates" / f"{name}.png"))
        if template is None:
            raise FileNotFoundError(f"缺少状态模板: {name}.png")
        STATE_TEMPLATE_CACHE[name] = template
    template = STATE_TEMPLATE_CACHE[name]
    geometry = BoardGeometry.from_image(image)
    x, y, width, height = STATE_REGIONS[name]
    x1, y1 = round(x * geometry.scale_x), round(y * geometry.scale_y)
    x2, y2 = round((x + width) * geometry.scale_x), round((y + height) * geometry.scale_y)
    roi = image[y1:y2, x1:x2]
    resized = cv2.resize(
        template,
        (
            max(8, round(template.shape[1] * geometry.scale_x)),
            max(8, round(template.shape[0] * geometry.scale_y)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    if roi.shape[0] < resized.shape[0] or roi.shape[1] < resized.shape[1]:
        return 0.0
    result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(result)
    return float(score)


class CellStabilizer:
    def __init__(self, required_frames: int) -> None:
        self.required_frames = max(1, required_frames)
        self.previous = np.full((ROWS, COLS), UNKNOWN, dtype=np.int8)
        self.counts = np.zeros((ROWS, COLS), dtype=np.int16)

    def reset(self) -> None:
        self.previous.fill(UNKNOWN)
        self.counts.fill(0)

    def update(self, reading: BoardReading) -> BoardReading:
        current = reading.labels
        known = current >= 0
        same = known & (current == self.previous)
        self.counts = np.where(same, self.counts + 1, np.where(known, 1, 0))
        self.previous = current.copy()
        stable = known & (self.counts >= self.required_frames)
        return BoardReading(
            labels=current,
            confidence=reading.confidence,
            margins=reading.margins,
            stable=stable,
        )


@dataclass
class BotConfig:
    detect_interval_ms: int = 100
    cell_stable_frames: int = 2
    swipe_duration_ms: int = 120
    post_swipe_delay_ms: int = 120
    min_swipe_interval_ms: int = 300
    threshold: float = 0.72
    dry_run: bool = False
    debug: bool = False


@dataclass
class CandyBot:
    config: BotConfig
    stop: bool = False
    started: bool = False
    start_attempted: bool = False
    classifier: TemplateClassifier | None = field(default=None, init=False)
    capture_shape: tuple[int, int] | None = field(default=None, init=False)
    stabilizer: CellStabilizer = field(init=False)
    listener: keyboard.Listener | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.stabilizer = CellStabilizer(self.config.cell_stable_frames)

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        if key == keyboard.Key.esc:
            self.stop = True
            return False
        return None

    def ensure_classifier(self, image: np.ndarray) -> TemplateClassifier:
        if self.classifier is None:
            self.classifier = load_classifier(image, self.config.threshold)
        shape = image.shape[:2]
        if self.capture_shape != shape:
            self.capture_shape = shape
            self.stabilizer.reset()
        return self.classifier

    def click_start(self, frame: WindowFrame) -> None:
        geometry = BoardGeometry.from_image(frame.image)
        x, y = frame.screen_point(
            630 * geometry.scale_x,
            1530 * geometry.scale_y,
        )
        pyautogui.click(x=x, y=y, button="left")
        print("[INFO] 已点击开始游戏，等待倒计时结束")

    def swipe(self, frame: WindowFrame, move: Move) -> None:
        geometry = BoardGeometry.from_image(frame.image)
        first_x, first_y = geometry.center(*move.first)
        second_x, second_y = geometry.center(*move.second)
        x1, y1 = frame.screen_point(first_x, first_y)
        x2, y2 = frame.screen_point(second_x, second_y)
        pyautogui.moveTo(x=x1, y=y1)
        pyautogui.mouseDown(button="left")
        try:
            time.sleep(0.05)
            pyautogui.dragTo(
                x=x2,
                y=y2,
                duration=self.config.swipe_duration_ms / 1000,
                button="left",
                mouseDownUp=False,
            )
        finally:
            pyautogui.mouseUp(button="left")

    def save_debug(
        self,
        frame: WindowFrame,
        reading: BoardReading,
        move: Move | None,
        capture_ms: float,
        recognize_ms: float,
    ) -> None:
        if not self.config.debug:
            return
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        output = frame.image.copy()
        geometry = BoardGeometry.from_image(output)
        radius = max(8, round(56 * geometry.scale_x))
        for row in range(ROWS):
            for col in range(COLS):
                if reading.stable[row, col]:
                    continue
                x, y = geometry.center(row, col)
                cv2.rectangle(
                    output,
                    (x - radius, y - radius),
                    (x + radius, y + radius),
                    (0, 0, 255),
                    2,
                )
        if move is not None:
            first = geometry.center(*move.first)
            second = geometry.center(*move.second)
            cv2.arrowedLine(output, first, second, (0, 255, 0), 4, tipLength=0.25)
        cv2.imwrite(str(debug_dir / "current.png"), output)
        print(
            f"[DEBUG] 截图 {capture_ms:.1f}ms，识别 {recognize_ms:.1f}ms，"
            f"稳定 {reading.stable.sum()}/{ROWS * COLS}，候选 {move}"
        )

    def run(self) -> None:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.daemon = True
        self.listener.start()
        print("[INFO] 辅助脚本已启动，按 ESC 停止")
        print("[INFO] 等待开始页面或有效棋盘，请保持游戏窗口可见")
        last_swipe = 0.0
        unreadable_since: float | None = None
        previous_labels: np.ndarray | None = None
        blocked_moves: set[tuple[Cell, Cell]] = set()
        try:
            while not self.stop:
                cycle_start = time.perf_counter()
                frame = capture_game_window()
                capture_ms = (time.perf_counter() - cycle_start) * 1000
                if state_score(frame.image, "success") >= 0.76:
                    print("[INFO] 检测到挑战成功，脚本停止")
                    break
                if (
                    not self.started
                    and not self.start_attempted
                    and state_score(frame.image, "start") >= 0.68
                ):
                    if self.config.dry_run:
                        print("[INFO] 当前是开始页面，dry-run 不点击开始按钮")
                        break
                    self.start_attempted = True
                    self.click_start(frame)
                    time.sleep(3.25)
                    self.stabilizer.reset()
                    continue
                recognize_start = time.perf_counter()
                classifier = self.ensure_classifier(frame.image)
                reading = self.stabilizer.update(classifier.recognize(frame.image))
                recognize_ms = (time.perf_counter() - recognize_start) * 1000
                if not self.started:
                    if reading.stable.sum() == ROWS * COLS:
                        self.started = True
                        print("[INFO] 棋盘已稳定，开始寻找交换")
                    else:
                        time.sleep(self.config.detect_interval_ms / 1000)
                        continue
                if previous_labels is None or not np.array_equal(
                    reading.labels, previous_labels
                ):
                    blocked_moves.clear()
                previous_labels = reading.labels.copy()
                moves = [
                    move
                    for move in find_moves(reading.labels, reading.stable)
                    if (move.first, move.second) not in blocked_moves
                ]
                best = moves[0] if moves else None
                self.save_debug(frame, reading, best, capture_ms, recognize_ms)
                if self.config.dry_run:
                    visible = np.where(reading.stable, reading.labels, UNKNOWN)
                    print(format_board(visible))
                    print(f"\n稳定 {reading.stable.sum()}/{ROWS * COLS} 格，合法交换 {len(moves)} 个")
                    if best is not None:
                        print(f"建议 {format_move(best)}")
                    break
                now = time.perf_counter()
                if reading.stable.sum() < 8:
                    if unreadable_since is None:
                        unreadable_since = now
                    elif now - unreadable_since > 8:
                        print("[WARN] 连续无法可靠识别棋盘，脚本停止")
                        break
                else:
                    unreadable_since = None
                if best is not None:
                    elapsed_ms = (now - last_swipe) * 1000
                    if elapsed_ms >= self.config.min_swipe_interval_ms:
                        tag = "SPECIAL" if best.special else "MOVE"
                        print(f"[{tag}] {format_move(best)}")
                        self.swipe(frame, best)
                        blocked_moves.add((best.first, best.second))
                        last_swipe = time.perf_counter()
                        time.sleep(self.config.post_swipe_delay_ms / 1000)
                elapsed = time.perf_counter() - cycle_start
                remaining = self.config.detect_interval_ms / 1000 - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("\n[INFO] 收到 Ctrl+C，脚本停止")
        except pyautogui.FailSafeException:
            print("\n[INFO] 鼠标进入左上角安全区，脚本停止")
        finally:
            if self.listener is not None:
                self.listener.stop()
            print("[INFO] 已退出")


def run_offline(image_path: Path, threshold: float) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    reading = load_classifier(image, threshold).recognize(image)
    print(format_board(reading.labels))
    known = reading.labels >= 0
    if known.any():
        print(
            f"\n识别 {known.sum()}/{ROWS * COLS} 格，"
            f"最低置信度 {reading.confidence[known].min():.3f}"
        )
    else:
        print("没有识别到棋子")
    moves = find_moves(reading.labels, reading.stable)
    if not moves:
        print("当前清晰区域没有确定可消除的交换")
        return
    print(f"共 {len(moves)} 个合法交换，前 5 个：")
    for move in moves[:5]:
        print(format_move(move))


def main() -> None:
    parser = argparse.ArgumentParser(description="杖剑传说糖果消消夜辅助脚本")
    parser.add_argument("--image", type=Path, help="离线识别指定游戏截图")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--detect-interval-ms", type=int, default=100)
    parser.add_argument("--cell-stable-frames", type=int, default=2)
    parser.add_argument("--swipe-duration-ms", type=int, default=120)
    parser.add_argument("--post-swipe-delay-ms", type=int, default=120)
    parser.add_argument("--min-swipe-interval-ms", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="只识别并建议，不操作鼠标")
    parser.add_argument("--debug", action="store_true", help="保存诊断截图并输出耗时")
    args = parser.parse_args()
    if args.image is not None:
        run_offline(args.image, args.threshold)
        return
    config = BotConfig(
        detect_interval_ms=args.detect_interval_ms,
        cell_stable_frames=args.cell_stable_frames,
        swipe_duration_ms=args.swipe_duration_ms,
        post_swipe_delay_ms=args.post_swipe_delay_ms,
        min_swipe_interval_ms=args.min_swipe_interval_ms,
        threshold=args.threshold,
        dry_run=args.dry_run,
        debug=args.debug,
    )
    CandyBot(config).run()


if __name__ == "__main__":
    main()
