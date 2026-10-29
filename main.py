from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROWS = 9
COLS = 7
UNKNOWN = -1
REFERENCE_WIDTH = 1262
REFERENCE_HEIGHT = 1880
REFERENCE_XS = np.array([242, 371, 500, 630, 759, 888, 1017], dtype=np.float64)
REFERENCE_YS = np.array(
    [505, 634, 764, 893, 1022, 1152, 1281, 1410, 1540], dtype=np.float64
)
PIECE_NAMES = ("pumpkin", "tree", "castle", "bat", "ghost")
DISPLAY_NAMES = {
    "pumpkin": "南瓜",
    "tree": "树桩",
    "castle": "城堡",
    "bat": "蝙蝠",
    "ghost": "幽灵",
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
    def __init__(
        self,
        template_dir: Path,
        geometry: BoardGeometry,
        threshold: float = 0.72,
        min_margin: float = 0.025,
    ) -> None:
        self.threshold = threshold
        self.min_margin = min_margin
        size = geometry.template_size
        self.templates: list[np.ndarray] = []
        for name in PIECE_NAMES:
            path = template_dir / f"{name}.png"
            template = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if template is None:
                raise FileNotFoundError(f"缺少棋子模板: {path}")
            self.templates.append(
                cv2.resize(template, (size, size), interpolation=cv2.INTER_AREA)
            )

    def classify_cell(self, cell: np.ndarray) -> tuple[int, float, float]:
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
        if geometry.template_size != self.templates[0].shape[0]:
            raise ValueError("分类器模板尺寸与图像不匹配")
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

    @property
    def rank(self) -> tuple[int, int, int, int, int, int]:
        lowest = max(self.first[0], self.second[0])
        return (
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
            while end < COLS and label >= 0 and int(board[row, end]) == label:
                end += 1
            if label >= 0 and end - start >= 3:
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
            while end < ROWS and label >= 0 and int(board[end, col]) == label:
                end += 1
            if label >= 0 and end - start >= 3:
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
    if not stable[first] or not stable[second] or labels[first] == labels[second]:
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


def load_classifier(image: np.ndarray, threshold: float) -> TemplateClassifier:
    return TemplateClassifier(
        ROOT / "images" / "templates",
        BoardGeometry.from_image(image),
        threshold=threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="糖果消消夜棋盘识别与交换求解")
    parser.add_argument("--image", type=Path, required=True, help="待识别的游戏截图")
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"无法读取图片: {args.image}")
    reading = load_classifier(image, args.threshold).recognize(image)
    print(format_board(reading.labels))
    known = reading.labels >= 0
    print(
        f"\n识别 {known.sum()}/{ROWS * COLS} 格，"
        f"最低置信度 {reading.confidence[known].min():.3f}"
        if known.any()
        else "没有识别到棋子"
    )
    moves = find_moves(reading.labels, reading.stable)
    if not moves:
        print("当前清晰区域没有确定可消除的交换")
        return
    print(f"共 {len(moves)} 个合法交换，前 5 个：")
    for move in moves[:5]:
        a = (move.first[0] + 1, move.first[1] + 1)
        b = (move.second[0] + 1, move.second[1] + 1)
        print(
            f"{a} -> {b}，立即消除 {move.immediate_count}，"
            f"确定总消除 {move.guaranteed_count}，最长匹配 {move.longest_run}"
        )


if __name__ == "__main__":
    main()
