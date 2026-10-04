"""Таймлайн в стиле Vegas: линейка, дорожки с клипами (событиями), заголовки дорожек."""
import copy
import math

import numpy as np
from PySide6.QtCore import QLineF, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QScrollBar,
                               QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget)

from engine import (MIN_CLIP, SR, Project, clip_envelope, column_peaks, db_to_lin,
                    short_effect_name)

TRACK_COLORS = ["#3f86d6", "#d6773f", "#4fae5a", "#b65bc2", "#cdb43c", "#3fb7b7", "#d4475f", "#7f7fd8"]
HEADER_W = 230
RULER_H = 38
TITLE_H = 15
EDGE_PX = 6
SNAP_PX = 10
CURSOR_GRAB_PX = 4
MIN_PPS, MAX_PPS = 0.5, 20000.0


def fmt_time(samples):
    s = max(samples, 0) / SR
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def _ruler_label(sec, step):
    m, s = divmod(sec, 60)
    h, m = divmod(int(m), 60)
    head = f"{h}:{m:02d}:" if h else f"{m}:"
    if step < 1:
        return head + f"{s:06.3f}".rstrip("0").rstrip(".")
    return head + f"{int(round(s)):02d}"


def tick_steps(pps):
    steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30,
             60, 120, 300, 600, 1800, 3600]
    major = next((s for s in steps if s * pps >= 90), steps[-1])
    return major, major / 5


# =========================================================================== состояние

class EditorState(QObject):
    changed = Signal()            # перерисовка (курсор, выделение, вид, правки клипов)
    structure_changed = Signal()  # поменялся набор дорожек
    seek_requested = Signal()     # курсор передвинули руками (перемотка во время игры)

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.pps = 60.0           # пикселей в секунду
        self.offset = 0.0         # секунда, видимая у левого края
        self.track_h = 100
        self.selected = set()
        self.track = 0
        self.cursor = 0
        self.range = None         # (a, b) выделенный временной диапазон
        self.playhead = None
        self.snap = True
        self.dirty = False         # есть несохранённые изменения
        self.undo_stack = []
        self.redo_stack = []

    # координаты
    def x_of(self, sample):
        return (sample / SR - self.offset) * self.pps

    def sample_of(self, x):
        return max(0, int(round((x / self.pps + self.offset) * SR)))

    def set_zoom(self, pps, anchor_x):
        t = anchor_x / self.pps + self.offset
        self.pps = min(max(pps, MIN_PPS), MAX_PPS)
        self.offset = max(0.0, t - anchor_x / self.pps)
        self.changed.emit()

    def snap_sample(self, s, exclude=(), use_cursor=True):
        if not self.snap:
            return s
        tol = SNAP_PX / self.pps * SR
        points = [0, self.cursor] if use_cursor else [0]
        if self.range:
            points += list(self.range)
        for c in self.project.clips:
            if c not in exclude:
                points += (c.start, c.end)
        best = min(points, key=lambda p: abs(p - s))
        return best if abs(best - s) <= tol else s

    # история
    def checkpoint(self):
        self.undo_stack.append(self.project.snapshot())
        del self.undo_stack[:-200]
        self.redo_stack.clear()
        self.dirty = True

    def _swap(self, src, dst):
        if not src:
            return False
        dst.append(self.project.snapshot())
        self.dirty = True
        self.project.restore(src.pop())
        self.selected.clear()
        self.track = min(self.track, max(len(self.project.tracks) - 1, 0))
        self.structure_changed.emit()
        self.changed.emit()
        return True

    def undo(self):
        return self._swap(self.undo_stack, self.redo_stack)

    def redo(self):
        return self._swap(self.redo_stack, self.undo_stack)


# =========================================================================== линейка

class Ruler(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setFixedHeight(RULER_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._anchor = None
        self._press_x = 0

    def paintEvent(self, _):
        st = self.state
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#2d2d30"))
        if st.range:
            xa, xb = st.x_of(st.range[0]), st.x_of(st.range[1])
            p.fillRect(QRectF(xa, 0, xb - xa, 9), QColor("#4f8fd8"))
            p.fillRect(QRectF(xa, 9, xb - xa, h - 9), QColor(79, 143, 216, 50))
        major, minor = tick_steps(st.pps)
        t0, t1 = st.offset, st.offset + w / st.pps
        p.setPen(QColor("#777"))
        k = math.floor(t0 / minor)
        while k * minor <= t1:
            x = (k * minor - t0) * st.pps
            p.drawLine(QLineF(x, h - 6, x, h))
            k += 1
        p.setPen(QColor("#bbb"))
        font = QFont(p.font())
        font.setPointSize(8)
        p.setFont(font)
        k = math.floor(t0 / major)
        while k * major <= t1:
            t = k * major
            x = (t - t0) * st.pps
            p.drawLine(QLineF(x, h - 14, x, h))
            p.drawText(QPointF(x + 3, h - 16), _ruler_label(t, major))
            k += 1
        p.setPen(QColor("#111"))
        p.drawLine(0, h - 1, w, h - 1)
        # курсор
        x = st.x_of(st.cursor)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#f0f0f0"))
        p.drawPolygon(QPolygonF([QPointF(x - 6, h - 11), QPointF(x + 6, h - 11), QPointF(x, h - 1)]))
        if st.playhead is not None:
            x = st.x_of(st.playhead)
            p.setPen(QPen(QColor("#ff5050"), 1.5))
            p.drawLine(QLineF(x, 0, x, h))
        p.end()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        st = self.state
        self._press_x = e.position().x()
        if abs(self._press_x - st.x_of(st.cursor)) <= CURSOR_GRAB_PX + 3:
            self._dragging_cursor = True      # тянем за треугольник курсора
            self._anchor = None
            return
        self._dragging_cursor = False
        self._anchor = st.snap_sample(st.sample_of(self._press_x))
        st.cursor = self._anchor
        st.changed.emit()

    def mouseMoveEvent(self, e):
        st = self.state
        x = e.position().x()
        if not e.buttons():
            near = abs(x - st.x_of(st.cursor)) <= CURSOR_GRAB_PX + 3
            self.setCursor(Qt.SplitHCursor if near else Qt.ArrowCursor)
            return
        if getattr(self, "_dragging_cursor", False):
            st.cursor = st.snap_sample(st.sample_of(x), use_cursor=False)
            st.changed.emit()
            return
        if self._anchor is None or abs(x - self._press_x) < 3:
            return
        s = st.snap_sample(st.sample_of(x))
        a, b = sorted((self._anchor, s))
        st.range = (a, b) if b > a else None
        st.cursor = a
        st.changed.emit()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.state.seek_requested.emit()
        self._anchor = None
        self._dragging_cursor = False

    def mouseDoubleClickEvent(self, e):
        self.state.range = None
        self.state.changed.emit()


# =========================================================================== холст

class TrackCanvas(QWidget):
    edited = Signal()
    context_menu_requested = Signal(object)
    clip_double_clicked = Signal(object)
    files_dropped = Signal(list, int, int)   # пути, индекс дорожки, позиция

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._mode = None
        self._clip = None
        self._orig = {}
        self._press = (0, 0, 0)
        self._moved = False
        self._anchor = 0
        self._auto = {}
        self._exposed = (0, 0)
        self._ctrl = False
        self._copied = False
        self._pending_deselect = None
        self._cursor_before = 0

    # ------------------------------------------------------------------ рисование
    def paintEvent(self, e):
        st = self.state
        proj = st.project
        self._exposed = (e.rect().left() - 1, e.rect().right() + 2)
        p = QPainter(self)
        w, h = self.width(), self.height()
        th = st.track_h
        p.fillRect(0, 0, w, h, QColor("#1b1b1c"))
        for i, t in enumerate(proj.tracks):
            y = i * th
            base = QColor("#2e3238") if i == st.track else QColor("#262628" if i % 2 else "#2a2a2c")
            p.fillRect(0, y, w, th, base)
            p.fillRect(0, y + th - 1, w, 1, QColor("#121212"))
        tracks_h = len(proj.tracks) * th
        # сетка
        major, _ = tick_steps(st.pps)
        p.setPen(QColor(255, 255, 255, 14))
        k = math.floor(st.offset / major)
        while (k * major - st.offset) * st.pps <= w:
            x = (k * major - st.offset) * st.pps
            p.drawLine(QLineF(x, 0, x, tracks_h))
            k += 1
        self._auto = proj.auto_fades()
        for c in proj.clips:
            if 0 <= c.track < len(proj.tracks):
                x0, x1 = st.x_of(c.start), st.x_of(c.end)
                if x1 >= self._exposed[0] and x0 <= self._exposed[1]:
                    self._draw_clip(p, c, x0, x1)
        if proj.auto_crossfade:
            for ca, cb in proj.overlaps():
                if 0 <= ca.track < len(proj.tracks):
                    self._draw_crossfade(p, ca, cb)
        if st.range:
            xa, xb = st.x_of(st.range[0]), st.x_of(st.range[1])
            p.fillRect(QRectF(xa, 0, xb - xa, max(tracks_h, h)), QColor(120, 170, 255, 38))
            p.setPen(QPen(QColor(120, 170, 255, 160), 1))
            p.drawLine(QLineF(xa, 0, xa, h))
            p.drawLine(QLineF(xb, 0, xb, h))
        x = st.x_of(st.cursor)
        p.setPen(QPen(QColor("#000"), 1))
        p.drawLine(QLineF(x, 0, x, h))
        p.setPen(QPen(QColor("#fff"), 1, Qt.DashLine))
        p.drawLine(QLineF(x, 0, x, h))
        if st.playhead is not None:
            x = st.x_of(st.playhead)
            p.setPen(QPen(QColor("#ff5050"), 1.5))
            p.drawLine(QLineF(x, 0, x, h))
        if not proj.clips:
            p.setPen(QColor("#666"))
            p.drawText(QRectF(0, tracks_h, w, max(h - tracks_h, 80)), Qt.AlignCenter,
                       "Перетащите сюда аудио/видео файлы или нажмите Файл → Импорт (Ctrl+I)")
        p.end()

    def _draw_clip(self, p, c, x0, x1):
        st = self.state
        track = st.project.tracks[c.track]
        col = QColor(TRACK_COLORS[track.color % len(TRACK_COLORS)])
        if track.mute:
            col = QColor(col.red() // 2 + 50, col.green() // 2 + 50, col.blue() // 2 + 50)
        sel = c in st.selected
        y = c.track * st.track_h + 1
        hh = st.track_h - 3
        rect = QRectF(x0, y, x1 - x0, hh)
        p.save()
        p.setClipRect(rect.intersected(QRectF(-2, 0, self.width() + 4, self.height())))
        p.fillRect(rect, col.darker(250 if not sel else 170))
        p.fillRect(QRectF(x0, y, x1 - x0, TITLE_H), col.lighter(125) if sel else col)
        p.setPen(QColor("#fff") if sel else QColor("#101010"))
        font = QFont(p.font())
        font.setPointSize(8)
        p.setFont(font)
        title = c.name
        if c.effects:
            names = [short_effect_name(fx) for fx in c.effects if fx.get("enabled", True)]
            title += "   [fx: " + (", ".join(names) or "выкл.") + "]"
        p.drawText(QRectF(max(x0, 0) + 4, y, max(x1 - max(x0, 0) - 8, 0), TITLE_H),
                   Qt.AlignVCenter | Qt.AlignLeft, title)
        wave = QRectF(x0, y + TITLE_H, x1 - x0, hh - TITLE_H)
        self._draw_wave(p, c, x0, x1, wave, col)
        # ручные фейды (автоматические кроссфейды рисуются отдельно поверх)
        fi = c.fade_in / SR * st.pps
        fo = c.fade_out / SR * st.pps
        top, bottom = wave.top(), wave.bottom()
        shade = QColor(0, 0, 0, 120)
        p.setPen(QPen(QColor("#e8e8e8"), 1))
        if fi > 0:
            p.setBrush(shade)
            p.drawPolygon(QPolygonF([QPointF(x0, top), QPointF(x0 + fi, top), QPointF(x0, bottom)]))
        if fo > 0:
            p.setBrush(shade)
            p.drawPolygon(QPolygonF([QPointF(x1, top), QPointF(x1 - fo, top), QPointF(x1, bottom)]))
        p.setBrush(QColor("#f0f0f0"))
        p.setPen(QPen(QColor("#000"), 1))
        for hx in (x0 + fi, x1 - fo):
            p.drawRect(QRectF(hx - 3, y + 1, 6, 6))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#fff"), 1.5) if sel else QPen(col.darker(300), 1))
        p.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        p.restore()

    def _draw_crossfade(self, p, a, b):
        """Зона кроссфейда: подсветка и две равномощные кривые (затухание a, нарастание b)."""
        st = self.state
        xa, xb = st.x_of(b.start), st.x_of(a.end)
        if xb < self._exposed[0] or xa > self._exposed[1] or xb - xa < 2:
            return
        top = a.track * st.track_h + 1 + TITLE_H
        bottom = a.track * st.track_h + st.track_h - 3
        hgt = bottom - top
        p.fillRect(QRectF(xa, top, xb - xa, hgt), QColor(255, 220, 120, 40))
        steps = max(8, min(64, int(xb - xa) // 3))
        out_path, in_path = QPainterPath(), QPainterPath()
        for i in range(steps + 1):
            t = i / steps
            x = xa + (xb - xa) * t
            y_out = bottom - hgt * math.cos(t * math.pi / 2)
            y_in = bottom - hgt * math.sin(t * math.pi / 2)
            if i == 0:
                out_path.moveTo(x, y_out)
                in_path.moveTo(x, y_in)
            else:
                out_path.lineTo(x, y_out)
                in_path.lineTo(x, y_in)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#ffd27a"), 1.5))
        p.drawPath(out_path)
        p.drawPath(in_path)
        p.setPen(QPen(QColor(255, 210, 122, 150), 1, Qt.DashLine))
        p.drawLine(QLineF(xa, top, xa, bottom))
        p.drawLine(QLineF(xb, top, xb, bottom))

    def _draw_wave(self, p, c, x0, x1, area, col):
        st = self.state
        xa = max(int(math.floor(x0)), 0, self._exposed[0])
        xb = min(int(math.ceil(x1)), self.width(), self._exposed[1])
        if xb - xa < 1 or area.height() < 8:
            return
        cols = np.arange(xa, xb, dtype=np.float64)
        spp = SR / st.pps
        rel = (cols / st.pps + st.offset) * SR - c.start
        rel = np.clip(rel, 0, c.length - 1)
        starts = (c.src_start + rel).astype(np.int64)
        end = int(min(c.src_start + c.length, starts[-1] + max(spp, 1)))
        if end <= starts[0]:
            return
        mn, mx, rms = column_peaks(c.source, starts, end)
        env = clip_envelope(c, rel + spp / 2, self._auto.get(c, (0, 0))) * db_to_lin(c.gain_db)
        env = env[:, None]
        mn = np.clip(mn * env, -1, 1)
        mx = np.clip(mx * env, -1, 1)
        rms = np.clip(rms * env, 0, 1)
        xs = (cols + 0.5).tolist()
        ch_h = area.height() / 2
        peak_col = col.lighter(150)
        rms_col = col.lighter(190)
        for k in range(2):
            center = area.top() + ch_h * (k + 0.5)
            amp = ch_h / 2 * 0.94
            y1 = (center - mx[:, k] * amp)
            y2 = np.maximum(center - mn[:, k] * amp, y1 + 1)
            p.setPen(QPen(peak_col, 1))
            p.drawLines([QLineF(x, a, x, b) for x, a, b in zip(xs, y1.tolist(), y2.tolist())])
            r = rms[:, k] * amp
            p.setPen(QPen(rms_col, 1))
            p.drawLines([QLineF(x, center - v, x, center + v)
                         for x, v in zip(xs, r.tolist()) if v >= 0.5])
            p.setPen(QPen(QColor(0, 0, 0, 90), 1))
            p.drawLine(QLineF(xa, center, xb, center))

    # ------------------------------------------------------------------ попадание
    def hit(self, x, y):
        st = self.state
        row = int(y // st.track_h)
        for c in reversed(st.project.clips):
            if c.track != row:
                continue
            x0, x1 = st.x_of(c.start), st.x_of(c.end)
            if not (x0 <= x <= x1):
                continue
            if y - row * st.track_h <= TITLE_H:
                if abs(x - (x0 + c.fade_in / SR * st.pps)) <= EDGE_PX:
                    return c, "fade_in"
                if abs(x - (x1 - c.fade_out / SR * st.pps)) <= EDGE_PX:
                    return c, "fade_out"
            if x1 - x0 > 3 * EDGE_PX:
                if x - x0 <= EDGE_PX:
                    return c, "trim_l"
                if x1 - x <= EDGE_PX:
                    return c, "trim_r"
            return c, "body"
        return None, None

    # ------------------------------------------------------------------ мышь
    def mousePressEvent(self, e):
        st = self.state
        x, y = e.position().x(), e.position().y()
        row = int(y // st.track_h)
        ntr = len(st.project.tracks)
        clip, zone = self.hit(x, y)
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        if 0 <= row < ntr:
            st.track = row
        if e.button() == Qt.RightButton:
            if clip is None:
                st.selected.clear()
            elif clip not in st.selected:
                st.selected = {clip}
            st.changed.emit()
            self.context_menu_requested.emit(e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        self._press = (x, y, row)
        self._moved = False
        self._ctrl = ctrl
        self._copied = False
        self._pending_deselect = None
        self._cursor_before = st.cursor
        if zone not in ("trim_l", "trim_r", "fade_in", "fade_out") and                 abs(x - st.x_of(st.cursor)) <= CURSOR_GRAB_PX:
            self._mode = "cursor"             # тянем курсор
            return
        if clip is not None:
            if ctrl and clip in st.selected:
                # снимем выделение при отпускании, если не будет Ctrl+перетаскивания (копии)
                self._pending_deselect = clip
            elif ctrl:
                st.selected.add(clip)
            elif clip not in st.selected:
                st.selected = {clip}
            self._mode = zone
            self._clip = clip
            group = st.selected if zone == "body" else {clip}
            self._orig = {c: (c.start, c.track, c.src_start, c.length, c.fade_in, c.fade_out)
                          for c in group}
        else:
            if not ctrl:
                st.selected.clear()
            st.range = None
            self._anchor = st.snap_sample(st.sample_of(x))
            st.cursor = self._anchor
            self._mode = "range"
        st.changed.emit()

    def mouseMoveEvent(self, e):
        st = self.state
        x, y = e.position().x(), e.position().y()
        if not (e.buttons() & Qt.LeftButton) or self._mode is None:
            _, zone = self.hit(x, y)
            if zone not in ("trim_l", "trim_r", "fade_in", "fade_out") and                     abs(x - st.x_of(st.cursor)) <= CURSOR_GRAB_PX:
                self.setCursor(Qt.SplitHCursor)
                return
            if zone == "body" and e.modifiers() & Qt.ControlModifier:
                self.setCursor(Qt.DragCopyCursor)
                return
            shapes = {"trim_l": Qt.SizeHorCursor, "trim_r": Qt.SizeHorCursor,
                      "fade_in": Qt.CrossCursor, "fade_out": Qt.CrossCursor,
                      "body": Qt.OpenHandCursor}
            self.setCursor(shapes.get(zone, Qt.ArrowCursor))
            return
        px, py, _ = self._press
        if not self._moved and abs(x - px) < 3 and abs(y - py) < 3:
            return
        if self._mode == "cursor":
            self._autoscroll(x)
            st.cursor = st.snap_sample(st.sample_of(x), use_cursor=False)
            st.changed.emit()
            return
        if not self._moved and self._mode != "range":
            st.checkpoint()
            if self._mode == "body" and self._ctrl:
                self._make_copies()
        self._moved = True
        self._autoscroll(x)
        if self._mode == "range":
            s = st.snap_sample(st.sample_of(x))
            a, b = sorted((self._anchor, s))
            st.range = (a, b) if b > a else None
            st.cursor = a
        elif self._mode == "body":
            self._drag_move(x, y)
        else:
            self._drag_edge(x)
        st.changed.emit()

    def _make_copies(self):
        """Ctrl+перетаскивание: на старом месте остаются копии, а тащим оригиналы."""
        st = self.state
        self._pending_deselect = None
        for c in self._orig:
            dup = copy.copy(c)
            st.project.clips.insert(st.project.clips.index(c), dup)
        self._copied = True
        self.setCursor(Qt.DragCopyCursor)

    def _drag_move(self, x, y):
        st = self.state
        c0 = self._clip
        orig = self._orig
        start0 = orig[c0][0]
        new = start0 + int(round((x - self._press[0]) / st.pps * SR))
        if st.snap:
            left = st.snap_sample(new, orig)
            right = st.snap_sample(new + c0.length, orig) - c0.length
            dl, dr = left - new, right - new
            if dl and (not dr or abs(dl) <= abs(dr)):
                new = left
            elif dr:
                new = right
        ds = max(new - start0, -min(v[0] for v in orig.values()))
        tracks = [v[1] for v in orig.values()]
        row = int(y // st.track_h)
        dt = row - self._press[2]
        dt = min(max(dt, -min(tracks)), len(st.project.tracks) - 1 - max(tracks))
        for c, v in orig.items():
            c.start = v[0] + ds
            c.track = v[1] + dt
        st.track = c0.track

    def _drag_edge(self, x):
        st = self.state
        c = self._clip
        s0, _, src0, len0, fi, fo = self._orig[c]
        pos = st.snap_sample(st.sample_of(x), {c})
        if self._mode == "trim_l":
            pos = min(max(pos, s0 - src0, 0), s0 + len0 - MIN_CLIP)
            d = pos - s0
            c.start, c.src_start, c.length = pos, src0 + d, len0 - d
            c.fade_in, c.fade_out = fi, fo
            c.clamp_fades()
        elif self._mode == "trim_r":
            pos = min(max(pos, s0 + MIN_CLIP), s0 + c.source.length - src0)
            c.length = pos - s0
            c.fade_in, c.fade_out = fi, fo
            c.clamp_fades()
        elif self._mode == "fade_in":
            c.fade_in = int(min(max(st.sample_of(x) - c.start, 0), c.length - c.fade_out))
        elif self._mode == "fade_out":
            c.fade_out = int(min(max(c.end - st.sample_of(x), 0), c.length - c.fade_in))

    def _autoscroll(self, x):
        st = self.state
        margin = 30
        if x > self.width() - margin:
            st.offset += (x - self.width() + margin) / st.pps * 0.2
        elif x < margin and st.offset > 0:
            st.offset = max(0.0, st.offset - (margin - x) / st.pps * 0.2)

    def mouseReleaseEvent(self, e):
        st = self.state
        if e.button() != Qt.LeftButton:
            return
        if self._mode == "body" and not self._moved:
            if self._pending_deselect is not None:
                st.selected.discard(self._pending_deselect)
            elif not self._ctrl:
                st.selected = {self._clip}
            st.cursor = st.sample_of(e.position().x())
        if self._moved and self._mode not in ("range", "cursor"):
            self.edited.emit()
        if st.cursor != self._cursor_before:
            st.seek_requested.emit()
        self._mode = None
        self._orig = {}
        st.changed.emit()

    def mouseDoubleClickEvent(self, e):
        clip, _ = self.hit(e.position().x(), e.position().y())
        if clip is not None:
            self._mode = None
            self.clip_double_clicked.emit(clip)

    def wheelEvent(self, e):
        st = self.state
        d = e.angleDelta().y() or e.angleDelta().x()
        mods = e.modifiers()
        if mods & Qt.ControlModifier:
            st.set_zoom(st.pps * 1.2 ** (d / 120), e.position().x())
            e.accept()
        elif mods & Qt.ShiftModifier or e.angleDelta().x():
            st.offset = max(0.0, st.offset - d / 120 * 100 / st.pps)
            st.changed.emit()
            e.accept()
        else:
            e.ignore()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.state.changed.emit()

    # ------------------------------------------------------------------ drag & drop файлов
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            pos = e.position()
            self.files_dropped.emit(paths, int(pos.y() // self.state.track_h),
                                    self.state.sample_of(pos.x()))
            e.acceptProposedAction()


# =========================================================================== заголовки дорожек

class NoWheelSlider(QSlider):
    def wheelEvent(self, e):
        e.ignore()


class TrackHeader(QFrame):
    def __init__(self, state, index, on_delete):
        super().__init__()
        self.state = state
        self.index = index
        self.track = state.project.tracks[index]
        self.setFixedSize(HEADER_W, state.track_h)
        self.setObjectName("TrackHeader")
        t = self.track
        color = TRACK_COLORS[t.color % len(TRACK_COLORS)]

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 4, 1)
        outer.setSpacing(4)
        strip = QFrame()
        strip.setFixedWidth(7)
        strip.setStyleSheet(f"background:{color};")
        outer.addWidget(strip)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(0, 3, 0, 3)
        outer.addLayout(col)

        row1 = QHBoxLayout()
        num = QLabel(str(index + 1))
        num.setFixedWidth(16)
        num.setStyleSheet("color:#aaa; font-weight:bold;")
        self.name = QLineEdit(t.name)
        self.name.setFrame(False)
        self.name.editingFinished.connect(self._rename)
        self.mute = self._button("M", "Заглушить", t.mute, self._toggle_mute)
        self.solo = self._button("S", "Соло", t.solo, self._toggle_solo)
        rm = self._button("✕", "Удалить дорожку", False, lambda _: on_delete(self.index), checkable=False)
        for wdg in (num, self.name, self.mute, self.solo, rm):
            row1.addWidget(wdg)
        col.addLayout(row1)

        self.vol = NoWheelSlider(Qt.Horizontal)
        self.vol.setRange(-600, 120)
        self.vol.setValue(int(round(t.volume_db * 10)))
        self.vol_lbl = QToolButton()
        self.vol_lbl.setToolTip("Громкость дорожки (клик — сброс в 0 дБ)")
        self.vol.sliderPressed.connect(state.checkpoint)
        self.vol.valueChanged.connect(self._vol_changed)
        self.vol_lbl.clicked.connect(lambda: self._reset(self.vol, 0))
        col.addLayout(self._slider_row("Гр.", self.vol, self.vol_lbl))

        self.pan = NoWheelSlider(Qt.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(int(round(t.pan * 100)))
        self.pan_lbl = QToolButton()
        self.pan_lbl.setToolTip("Панорама (клик — по центру)")
        self.pan.sliderPressed.connect(state.checkpoint)
        self.pan.valueChanged.connect(self._pan_changed)
        self.pan_lbl.clicked.connect(lambda: self._reset(self.pan, 0))
        col.addLayout(self._slider_row("Пан.", self.pan, self.pan_lbl))
        self._vol_changed(self.vol.value())
        self._pan_changed(self.pan.value())
        for w in self.findChildren(QWidget):
            if w is not self.name:
                w.setFocusPolicy(Qt.NoFocus)
        self.set_selected(index == state.track)

    def _button(self, text, tip, checked, slot, checkable=True):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setFixedSize(22, 20)
        b.setCheckable(checkable)
        b.setChecked(checked)
        b.setObjectName("trackBtn" + text)
        b.clicked.connect(slot)
        return b

    def _slider_row(self, title, slider, label):
        row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFixedWidth(28)
        lbl.setStyleSheet("color:#999; font-size:8pt;")
        label.setFixedWidth(58)
        label.setAutoRaise(True)
        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(label)
        return row

    def _reset(self, slider, value):
        self.state.checkpoint()
        slider.setValue(value)

    def _rename(self):
        if self.name.text() != self.track.name:
            self.state.checkpoint()
            self.track.name = self.name.text()

    def _toggle_mute(self, on):
        self.state.checkpoint()
        self.track.mute = on
        self.state.changed.emit()

    def _toggle_solo(self, on):
        self.state.checkpoint()
        self.track.solo = on
        self.state.changed.emit()

    def _vol_changed(self, v):
        self.track.volume_db = v / 10
        self.vol_lbl.setText("-inf дБ" if v <= -600 else f"{v / 10:+.1f} дБ")
        if v <= -600:
            self.track.volume_db = -200.0

    def _pan_changed(self, v):
        self.track.pan = v / 100
        self.pan_lbl.setText("Центр" if v == 0 else (f"{-v}% L" if v < 0 else f"{v}% R"))

    def set_selected(self, on):
        self.setProperty("selected", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        self.state.track = self.index
        self.state.changed.emit()


# =========================================================================== контейнер

class TimelineWidget(QWidget):
    def __init__(self, state, on_delete_track):
        super().__init__()
        self.state = state
        self.on_delete_track = on_delete_track
        self.headers = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        top = QHBoxLayout()
        top.setSpacing(0)
        self.time_box = QLabel()
        self.time_box.setObjectName("TimeBox")
        self.time_box.setFixedSize(HEADER_W, RULER_H)
        self.time_box.setAlignment(Qt.AlignCenter)
        self.ruler = Ruler(state)
        top.addWidget(self.time_box)
        top.addWidget(self.ruler)
        lay.addLayout(top)

        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        hb = QHBoxLayout(body)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        self.header_col = QWidget()
        self.header_col.setObjectName("HeaderColumn")
        self.header_col.setFixedWidth(HEADER_W)
        self.header_lay = QVBoxLayout(self.header_col)
        self.header_lay.setContentsMargins(0, 0, 0, 0)
        self.header_lay.setSpacing(0)
        self.canvas = TrackCanvas(state)
        hb.addWidget(self.header_col)
        hb.addWidget(self.canvas)
        self.body = body
        self.scroll.setWidget(body)
        lay.addWidget(self.scroll, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(0)
        self.zoom_lbl = QLabel()
        self.zoom_lbl.setFixedWidth(HEADER_W)
        self.zoom_lbl.setStyleSheet("color:#888; padding-left:6px; font-size:8pt;")
        self.hbar = QScrollBar(Qt.Horizontal)
        self.hbar.valueChanged.connect(self._hbar_moved)
        bottom.addWidget(self.zoom_lbl)
        bottom.addWidget(self.hbar)
        lay.addLayout(bottom)

        state.changed.connect(self.refresh)
        state.structure_changed.connect(self.rebuild)
        self.rebuild()

    def rebuild(self):
        while self.header_lay.count():
            item = self.header_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.headers = [TrackHeader(self.state, i, self.on_delete_track)
                        for i in range(len(self.state.project.tracks))]
        for h in self.headers:
            self.header_lay.addWidget(h)
        self.header_lay.addStretch(1)
        self.body.setMinimumHeight(len(self.headers) * self.state.track_h + 120)
        self.refresh()

    def refresh(self):
        st = self.state
        self.canvas.update()
        self.ruler.update()
        for i, h in enumerate(self.headers):
            if h.property("selected") != (i == st.track):
                h.set_selected(i == st.track)
        pos = st.playhead if st.playhead is not None else st.cursor
        self.time_box.setText(fmt_time(pos))
        self.zoom_lbl.setText(f"Масштаб: {st.pps:.1f} px/с" + ("   Привязка" if st.snap else ""))
        visible = max(self.canvas.width(), 1) / st.pps
        total = max(st.project.end() / SR + visible * 0.5, st.offset + visible)
        self.hbar.blockSignals(True)
        self.hbar.setRange(0, int(max(total - visible, 0) * 100))
        self.hbar.setPageStep(int(visible * 100))
        self.hbar.setSingleStep(max(int(visible * 10), 1))
        self.hbar.setValue(int(st.offset * 100))
        self.hbar.blockSignals(False)

    def _hbar_moved(self, v):
        self.state.offset = v / 100
        self.state.changed.emit()

    def visible_width(self):
        return max(self.canvas.width(), 1)

    def update_playhead(self, old, new):
        """Дешёвая перерисовка только полосок вокруг старого и нового положения плейхеда."""
        for s in (old, new):
            if s is not None:
                x = int(self.state.x_of(s))
                self.canvas.update(x - 3, 0, 7, self.canvas.height())
        self.ruler.update()
        self.time_box.setText(fmt_time(new))
