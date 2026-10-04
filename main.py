#!/usr/bin/env python3
"""Мини аудиоредактор в стиле Sony Vegas Pro (PySide6 + ffmpeg)."""
import copy
import os
import sys
import threading

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import (QAction, QColor, QKeySequence, QLinearGradient, QPainter, QPalette)
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QMenu, QMessageBox, QProgressDialog, QStyle, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

import engine
from dialogs import (ClipPropertiesDialog, EffectsManagerDialog, EQDialog, ExportDialog,
                     NoiseDialog, PitchDialog, ReverbDialog, VolumeDialog)
from engine import (CHANNEL_MODES, IMPORT_EXTS, PROJECT_EXT, SR, Clip, FFmpegError, Project,
                    Source, make_effect)
from player import Player
from timeline import EditorState, TimelineWidget, fmt_time


class LevelMeter(QWidget):
    """Индикатор уровня мастер-шины (L/R), -60…0 дБFS с удержанием пика."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 30)
        self.db = np.full(2, -60.0)
        self.hold = np.full(2, -60.0)

    def set_levels(self, lin):
        db = 20 * np.log10(np.maximum(np.asarray(lin, dtype=float), 1e-6))
        self.db = np.maximum(np.clip(db, -60, 0), self.db - 1.2)
        self.hold = np.maximum(self.db, self.hold - 0.3)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#111"))
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor("#1fa83a"))
        grad.setColorAt(0.75, QColor("#c8d42a"))
        grad.setColorAt(0.92, QColor("#e08a1e"))
        grad.setColorAt(1.0, QColor("#e02a2a"))
        bar_h = (h - 6) / 2
        for k in range(2):
            y = 2 + k * (bar_h + 2)
            frac = (self.db[k] + 60) / 60
            p.fillRect(QRectF(0, y, w * frac, bar_h), grad)
            hx = w * (self.hold[k] + 60) / 60
            p.fillRect(QRectF(hx - 1, y, 2, bar_h), QColor("#fff"))
        p.setPen(QColor(0, 0, 0, 160))
        for db in (-48, -36, -24, -12, -6, -3):
            x = w * (db + 60) / 60
            p.drawLine(int(x), 0, int(x), h)
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Vegas Audio")
        self.resize(1400, 820)
        self.state = EditorState()
        self.state.project.add_track()
        self.player = Player(lambda: self.state.project)
        self.clipboard = []
        self._busy = False
        self.project_path = None

        self.timeline = TimelineWidget(self.state, self.delete_track)
        canvas = self.timeline.canvas
        canvas.context_menu_requested.connect(self.show_context_menu)
        canvas.clip_double_clicked.connect(self.clip_properties)
        canvas.files_dropped.connect(self.import_files)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.timeline, 1)
        v.addWidget(self._build_transport())
        self.setCentralWidget(central)

        self.state.changed.connect(self._update_info)
        self.state.seek_requested.connect(self._on_seek)
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        backend = {"sounddevice": "sounddevice", "ffplay": "ffplay (упрощённый режим)",
                   None: "недоступен!"}[self.player.backend]
        self.statusBar().showMessage(f"Вывод звука: {backend}.  Частота проекта: {SR} Гц.")
        QTimer.singleShot(0, self._check_ffmpeg)

    # ------------------------------------------------------------------ построение UI
    def _act(self, text, slot, shortcut=None, icon=None, checkable=False, tip=None):
        a = QAction(text, self)
        if shortcut:
            seqs = list(shortcut) if isinstance(shortcut, (list, tuple)) else [shortcut]
            # однобуквенные клавиши дублируем для русской раскладки (W -> Ц и т.д.)
            seqs += [CYRILLIC[k] for k in seqs if k in CYRILLIC]
            a.setShortcuts([QKeySequence(k) for k in seqs])
        if icon is not None:
            a.setIcon(self.style().standardIcon(icon))
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_actions(self):
        S = QStyle.StandardPixmap
        A = self._act
        self.a_new = A("Новый проект", self.new_project, "Ctrl+N", S.SP_FileIcon)
        self.a_open = A("Открыть проект…", lambda: self.open_project(), "Ctrl+O", S.SP_DirOpenIcon)
        self.a_save = A("Сохранить проект", self.save, "Ctrl+S", S.SP_DialogSaveButton)
        self.a_save_as = A("Сохранить проект как…", self.save_as, "Ctrl+Shift+S")
        self.a_import = A("Импорт аудио/видео…", self.import_dialog, "Ctrl+I", S.SP_DialogOpenButton)
        self.a_export = A("Экспорт аудио…", self.export_dialog, ["Ctrl+E", "Ctrl+M"], S.SP_DriveHDIcon)
        self.a_quit = A("Выход", self.close, "Ctrl+Q")

        self.a_undo = A("Отменить", self.undo, "Ctrl+Z", S.SP_ArrowBack)
        self.a_redo = A("Повторить", self.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], S.SP_ArrowForward)
        self.a_cut = A("Вырезать", self.cut, "Ctrl+X")
        self.a_copy = A("Копировать", self.copy, "Ctrl+C")
        self.a_paste = A("Вставить в позицию курсора", self.paste, "Ctrl+V")
        self.a_delete = A("Удалить", self.delete, "Delete", S.SP_TrashIcon)
        self.a_split = A("Разрезать по курсору", self.split, "S", tip="Разрезать (S)")
        self.a_ripple = A("Удалить диапазон со сдвигом", self.ripple_delete, "Ctrl+Delete")
        self.a_select_all = A("Выделить всё", self.select_all, "Ctrl+A")
        self.a_props = A("Свойства клипа…", lambda: self.clip_properties(None), "Alt+Return")
        self.a_silence = A("Вставить пустой клип (тишина)…", self.insert_silence, "Ctrl+Shift+N")

        self.a_add_track = A("Добавить дорожку", self.add_track, "Ctrl+T")
        self.a_del_track = A("Удалить выбранную дорожку", lambda: self.delete_track(self.state.track))

        self.a_fx_volume = A("Громкость / нормализация…", self.fx_volume)
        self.a_fx_pitch = A("Высота тона и темп…", self.fx_pitch)
        self.a_fx_reverb = A("Реверберация…", self.fx_reverb)
        self.a_fx_eq = A("Эквалайзер…", self.fx_eq)
        self.a_fx_noise = A("Шум…", self.fx_noise)
        self.a_fx_reverse = A("Реверс", self.fx_reverse)
        self.a_fx_manager = A("Активные эффекты клипа…", lambda: self.effects_manager(), "Ctrl+Shift+F")
        self.a_ch_split = A("Разделить на левый и правый (две дорожки)", self.split_channels)
        self.a_ch_merge = A("Объединить два клипа в стерео (Л + П)", self.merge_channels)
        self.a_ch_modes = [A(title, lambda _=False, m=mode: self.fx_channels(m))
                           for mode, title in CHANNEL_MODES.items()]

        self.a_zoom_in = A("Увеличить", lambda: self.zoom(1.5), ["=", "+", "Ctrl+="], S.SP_ArrowUp)
        self.a_zoom_out = A("Уменьшить", lambda: self.zoom(1 / 1.5), ["-", "Ctrl+-"], S.SP_ArrowDown)
        self.a_zoom_fit = A("Показать весь проект", self.zoom_fit, "Ctrl+F")
        self.a_taller = A("Выше дорожки", lambda: self.track_height(20), "Ctrl+Shift+Up")
        self.a_shorter = A("Ниже дорожки", lambda: self.track_height(-20), "Ctrl+Shift+Down")
        self.a_snap = A("Привязка (snap)", self.toggle_snap, "N", checkable=True)
        self.a_snap.setChecked(True)
        self.a_xfade = A("Автоматический кроссфейд при наложении", self.toggle_crossfade, checkable=True)
        self.a_xfade.setChecked(True)

        self.a_play = A("Воспроизвести / стоп", self.play_stop, "Space", S.SP_MediaPlay)
        self.a_pause = A("Пауза", self.pause, ["Return", "Enter"], S.SP_MediaPause)
        self.a_stop = A("Стоп", self.stop, None, S.SP_MediaStop)
        self.a_home = A("В начало (W)", self.go_start, ["W", "Home"], S.SP_MediaSkipBackward)
        self.a_end = A("В конец", self.go_end, "End", S.SP_MediaSkipForward)
        self.a_loop = A("Цикл по выделению", lambda: None, "Q", S.SP_BrowserReload, checkable=True)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&Файл")
        for a in (self.a_new, self.a_open, self.a_save, self.a_save_as, None, self.a_import,
                  self.a_export, None, self.a_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Правка")
        for a in (self.a_undo, self.a_redo, None, self.a_cut, self.a_copy, self.a_paste, self.a_delete,
                  None, self.a_split, self.a_ripple, self.a_select_all, None, self.a_silence,
                  self.a_props):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Дорожка")
        m.addAction(self.a_add_track)
        m.addAction(self.a_del_track)
        self.fx_menu = mb.addMenu("&Эффекты")
        for a in (self.a_fx_volume, self.a_fx_pitch, self.a_fx_reverb, self.a_fx_eq,
                  self.a_fx_noise, self.a_fx_reverse):
            self.fx_menu.addAction(a)
        self.ch_menu = self.fx_menu.addMenu("Каналы")
        self.ch_menu.addAction(self.a_ch_split)
        self.ch_menu.addAction(self.a_ch_merge)
        self.ch_menu.addSeparator()
        for a in self.a_ch_modes:
            self.ch_menu.addAction(a)
        self.fx_menu.addSeparator()
        self.fx_menu.addAction(self.a_fx_manager)
        m = mb.addMenu("&Вид")
        for a in (self.a_zoom_in, self.a_zoom_out, self.a_zoom_fit, None, self.a_taller,
                  self.a_shorter, None, self.a_snap, self.a_xfade):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Транспорт")
        for a in (self.a_play, self.a_pause, self.a_stop, self.a_home, self.a_end, self.a_loop):
            m.addAction(a)
        m = mb.addMenu("&Справка")
        m.addAction(self._act("Горячие клавиши", self.show_help, "F1"))

    def _build_toolbar(self):
        tb = QToolBar("Инструменты")
        short = {self.a_new: "Новый", self.a_open: "Открыть", self.a_save: "Сохранить",
                 self.a_import: "Импорт", self.a_export: "Экспорт", self.a_undo: "Отменить",
                 self.a_redo: "Повторить", self.a_split: "Разрезать", self.a_delete: "Удалить",
                 self.a_zoom_in: "", self.a_zoom_out: "", self.a_snap: "Привязка"}
        for act, text in short.items():
            act.setIconText(text or " ")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_new, self.a_open, self.a_save, self.a_import, self.a_export, None,
                  self.a_undo, self.a_redo, None,
                  self.a_split, self.a_delete, None, self.a_zoom_in, self.a_zoom_out, self.a_snap):
            tb.addSeparator() if a is None else tb.addAction(a)
        fx = QToolButton()
        fx.setText("Эффекты")
        fx.setMenu(self.fx_menu)
        fx.setPopupMode(QToolButton.InstantPopup)
        tb.addSeparator()
        tb.addWidget(fx)
        self.addToolBar(tb)

    def _build_transport(self):
        bar = QWidget()
        bar.setObjectName("Transport")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        for a in (self.a_home, self.a_play, self.a_pause, self.a_stop, self.a_end, self.a_loop):
            b = QToolButton()
            b.setDefaultAction(a)
            b.setFocusPolicy(Qt.NoFocus)
            b.setIconSize(b.iconSize() * 1.3)
            h.addWidget(b)
        h.addSpacing(16)
        self.info = QLabel()
        self.info.setStyleSheet("color:#bbb;")
        h.addWidget(self.info, 1)
        h.addWidget(QLabel("Master"))
        self.meter = LevelMeter()
        h.addWidget(self.meter)
        return bar

    def _update_info(self):
        st = self.state
        parts = [f"Курсор: {fmt_time(st.cursor)}", f"Длина проекта: {fmt_time(st.project.end())}"]
        if st.range:
            a, b = st.range
            parts.append(f"Выделение: {fmt_time(a)} – {fmt_time(b)} ({(b - a) / SR:.3f} с)")
        if st.selected:
            parts.append(f"Клипов выделено: {len(st.selected)}")
        self.info.setText("    ".join(parts))
        name = os.path.basename(self.project_path) if self.project_path else "Без имени"
        self.setWindowTitle(f"{name}{' *' if st.dirty else ''} — Mini Vegas Audio")

    def _check_ffmpeg(self):
        try:
            engine.ffmpeg_bin()
        except FFmpegError as e:
            QMessageBox.critical(self, "ffmpeg", str(e))

    # ------------------------------------------------------------------ фоновые задачи
    def run_async(self, label, fn, on_done, cancellable=False):
        """Выполняет fn(progress) в потоке, показывая модальный прогресс."""
        if self._busy:
            return
        self._busy = True
        dlg = QProgressDialog(label, "Отмена" if cancellable else None, 0, 0, self)
        dlg.setWindowTitle("Подождите")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(250)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        if not cancellable:
            dlg.setCancelButton(None)
        shared = {"progress": None, "cancel": False}

        def progress(frac):
            shared["progress"] = frac
            return not shared["cancel"]

        def work():
            try:
                shared["result"] = fn(progress)
            except Exception as e:  # noqa: BLE001 — показываем пользователю любую ошибку
                shared["error"] = e

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        timer = QTimer(self)

        def poll():
            if dlg.wasCanceled():
                shared["cancel"] = True
            if shared["progress"] is not None:
                dlg.setMaximum(1000)
                dlg.setValue(int(shared["progress"] * 1000))
            if thread.is_alive():
                return
            timer.stop()
            timer.deleteLater()
            dlg.close()
            dlg.deleteLater()
            self._busy = False
            if "error" in shared:
                QMessageBox.critical(self, "Ошибка", str(shared["error"]))
            else:
                on_done(shared.get("result"))

        timer.timeout.connect(poll)
        timer.start(40)

    # ------------------------------------------------------------------ файл
    def new_project(self):
        if not self.maybe_save():
            return
        self.stop()
        project = Project()
        project.add_track()
        self._set_project(project, None)

    def _set_project(self, project, path, view=None):
        st = self.state
        st.project = project
        if not project.tracks:
            project.add_track()
        st.selected.clear()
        st.undo_stack.clear()
        st.redo_stack.clear()
        view = view or {}
        st.cursor = int(view.get("cursor", 0))
        st.offset = float(view.get("offset", 0.0))
        st.pps = float(view.get("pps", st.pps))
        st.track_h = int(view.get("track_h", st.track_h))
        st.range, st.track = None, 0
        st.dirty = False
        self.project_path = path
        self.a_xfade.setChecked(project.auto_crossfade)
        st.structure_changed.emit()
        self._update_info()

    def maybe_save(self):
        """Спрашивает про несохранённые изменения. False — пользователь передумал."""
        st = self.state
        if not st.dirty or not st.project.clips:
            return True
        ans = QMessageBox.question(
            self, "Несохранённые изменения", "Сохранить изменения в проекте?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if ans == QMessageBox.Save:
            return self.save()
        return ans == QMessageBox.Discard

    def open_project(self, path=None):
        if not self.maybe_save():
            return
        if path is None:
            folder = os.path.dirname(self.project_path) if self.project_path else os.path.expanduser("~")
            path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", folder,
                                                  f"Проект Mini Vegas (*.{PROJECT_EXT});;Все файлы (*)")
            if not path:
                return

        def done(result):
            project, view, errors = result
            self.stop()
            self._set_project(project, path, view)
            self.statusBar().showMessage(f"Открыт проект: {path}", 5000)
            if errors:
                QMessageBox.warning(self, "Открытие проекта",
                                    "Часть звука не загружена (клипы пропущены):\n\n" + "\n".join(errors))

        self.run_async("Открытие проекта…", lambda progress: engine.load_project(path, progress), done)

    def save(self):
        if not self.project_path:
            return self.save_as()
        return self._save_to(self.project_path)

    def save_as(self):
        start = self.project_path or os.path.join(os.path.expanduser("~"), f"project.{PROJECT_EXT}")
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", start,
                                              f"Проект Mini Vegas (*.{PROJECT_EXT})")
        if not path:
            return False
        if not path.lower().endswith("." + PROJECT_EXT):
            path += "." + PROJECT_EXT
        return self._save_to(path)

    def _save_to(self, path):
        st = self.state
        view = {"cursor": st.cursor, "offset": st.offset, "pps": st.pps, "track_h": st.track_h}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            engine.save_project(st.project, path, view)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Сохранение", f"Не удалось сохранить проект:\n{e}")
            return False
        QApplication.restoreOverrideCursor()
        self.project_path = path
        st.dirty = False
        self._update_info()
        self.statusBar().showMessage(f"Проект сохранён: {path}", 5000)
        return True

    def import_dialog(self):
        exts = " ".join(f"*.{e}" for e in IMPORT_EXTS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Импорт", os.path.expanduser("~"),
            f"Аудио и видео ({exts});;MP3 (*.mp3);;OGG (*.ogg *.oga);;WAV (*.wav);;MP4 (*.mp4 *.m4a);;Все файлы (*)")
        if paths:
            self.import_files(paths, self.state.track, self.state.cursor)

    def import_files(self, paths, track, pos):
        def work(progress):
            sources, errors = [], []
            for i, path in enumerate(paths):
                try:
                    sources.append(Source(engine.decode_file(path), os.path.basename(path), path))
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{os.path.basename(path)}: {e}")
                progress((i + 1) / len(paths))
            return sources, errors

        def done(result):
            sources, errors = result
            if errors:
                QMessageBox.warning(self, "Импорт", "Не удалось открыть:\n\n" + "\n\n".join(errors))
            if not sources:
                return
            st = self.state
            st.checkpoint()
            added_track = False
            tr = track
            if tr < 0 or tr >= len(st.project.tracks):
                tr = st.project.add_track()
                added_track = True
            p = pos
            new = []
            for s in sources:
                new.append(st.project.add_clip(s, tr, p))
                p += s.length
            st.selected = set(new)
            st.track = tr
            if added_track:
                st.structure_changed.emit()
            st.changed.emit()
            self.statusBar().showMessage(f"Импортировано файлов: {len(sources)}", 5000)

        self.run_async("Декодирование через ffmpeg и построение пиков…", work, done)

    def export_dialog(self):
        st = self.state
        if not st.project.clips:
            QMessageBox.information(self, "Экспорт", "Проект пуст — сначала импортируйте аудио.")
            return
        first = next((c.source.path for c in st.project.clips if c.source.path), None)
        folder = os.path.dirname(first) if first else os.path.expanduser("~")
        dlg = ExportDialog(self, folder, st.range is not None)
        if dlg.exec() != ExportDialog.Accepted:
            return
        v = dlg.values()
        start, end = st.range if v.pop("use_range") and st.range else (0, st.project.end())
        project = st.project

        def work(progress):
            return engine.export_project(project, start=start, end=end, progress=progress, **v)

        def done(ok):
            if ok:
                self.statusBar().showMessage(f"Экспорт завершён: {v['path']}", 10000)
                QMessageBox.information(self, "Экспорт", f"Готово:\n{v['path']}")
            else:
                self.statusBar().showMessage("Экспорт отменён", 5000)

        self.run_async(f"Экспорт в {os.path.basename(v['path'])}…", work, done, cancellable=True)

    # ------------------------------------------------------------------ правка
    def _selected_clips(self):
        return [c for c in self.state.project.clips if c in self.state.selected]

    def undo(self):
        if not self.state.undo():
            self.statusBar().showMessage("Нечего отменять", 2000)

    def redo(self):
        if not self.state.redo():
            self.statusBar().showMessage("Нечего повторять", 2000)

    def copy(self):
        clips = self._selected_clips()
        if not clips:
            return
        t0 = min(c.start for c in clips)
        tr0 = min(c.track for c in clips)
        self.clipboard = [(copy.copy(c), c.start - t0, c.track - tr0) for c in clips]
        self.statusBar().showMessage(f"Скопировано клипов: {len(clips)}", 3000)

    def cut(self):
        if self._selected_clips():
            self.copy()
            self.delete()

    def paste(self):
        st = self.state
        if not self.clipboard:
            return
        st.checkpoint()
        base = st.track if st.project.tracks else st.project.add_track()
        need = base + max(rt for _, _, rt in self.clipboard)
        added = False
        while len(st.project.tracks) <= need:
            st.project.add_track()
            added = True
        new = []
        for clip, rel, rt in self.clipboard:
            c = copy.copy(clip)
            c.start = st.cursor + rel
            c.track = base + rt
            st.project.clips.append(c)
            new.append(c)
        st.selected = set(new)
        st.cursor = max(c.end for c in new)
        if added:
            st.structure_changed.emit()
        st.changed.emit()

    def delete(self):
        st = self.state
        clips = self._selected_clips()
        if not clips:
            if st.range:
                self.ripple_delete()
            return
        st.checkpoint()
        st.project.clips = [c for c in st.project.clips if c not in st.selected]
        st.selected.clear()
        st.changed.emit()

    def split(self):
        st = self.state
        pos = st.cursor
        targets = self._selected_clips() or list(st.project.clips)
        targets = [c for c in targets if c.start < pos < c.end]
        if not targets:
            self.statusBar().showMessage("Под курсором нет клипов для разрезания", 3000)
            return
        st.checkpoint()
        rights = [r for r in (st.project.split_clip(c, pos) for c in targets) if r is not None]
        st.selected = set(rights)
        st.changed.emit()

    def ripple_delete(self):
        st = self.state
        if not st.range:
            self.statusBar().showMessage("Сначала выделите диапазон на линейке или пустом месте дорожки", 4000)
            return
        st.checkpoint()
        a, b = st.range
        st.project.delete_range(a, b)
        st.range = None
        st.cursor = a
        st.selected.clear()
        st.changed.emit()

    def select_all(self):
        self.state.selected = set(self.state.project.clips)
        self.state.changed.emit()

    def clip_properties(self, clip=None):
        st = self.state
        if clip is None:
            clips = self._selected_clips()
            if not clips:
                return
            clip = clips[0]
        dlg = ClipPropertiesDialog(self, clip)
        if dlg.exec() != ClipPropertiesDialog.Accepted:
            return
        st.checkpoint()
        v = dlg.values()
        clip.name = v["name"]
        clip.gain_db = v["gain_db"]
        clip.fade_in, clip.fade_out = v["fade_in"], v["fade_out"]
        clip.clamp_fades()
        st.changed.emit()

    def show_context_menu(self, pos):
        m = QMenu(self)
        if self.state.selected:
            for a in (self.a_split, self.a_cut, self.a_copy, self.a_delete):
                m.addAction(a)
            m.addSeparator()
            m.addMenu(self.fx_menu)
            m.addMenu(self.ch_menu)
            m.addAction(self.a_fx_manager)
            m.addAction(self.a_props)
        else:
            for a in (self.a_paste, self.a_import, self.a_silence, None, self.a_add_track,
                      self.a_ripple):
                m.addSeparator() if a is None else m.addAction(a)
        m.exec(pos)

    # ------------------------------------------------------------------ дорожки
    def add_track(self):
        st = self.state
        st.checkpoint()
        st.track = st.project.add_track()
        st.structure_changed.emit()

    def delete_track(self, index):
        st = self.state
        if not (0 <= index < len(st.project.tracks)):
            return
        if any(c.track == index for c in st.project.clips) and QMessageBox.question(
                self, "Удалить дорожку", "На дорожке есть клипы. Удалить её вместе с ними?") \
                != QMessageBox.Yes:
            return
        st.checkpoint()
        st.project.remove_track(index)
        st.selected = {c for c in st.selected if c in st.project.clips}
        st.track = max(0, min(st.track, len(st.project.tracks) - 1))
        st.structure_changed.emit()

    # ------------------------------------------------------------------ эффекты
    def process_clips(self, jobs, title, after=None):
        """jobs: [(клип, новая цепочка эффектов)]. Звук пересчитывается в фоне от исходника,
        затем цепочки применяются (с точкой отмены) и вызывается after()."""
        prepared = [(c, list(fx), c.base()) for c, fx in jobs]

        def work(progress):
            cache, out = {}, []
            for i, (_, effects, (base, bs, bl)) in enumerate(prepared):
                if not effects:
                    out.append(None)
                else:
                    key = engine.effects_key(base, bs, bl, effects)
                    if key not in cache:
                        data = engine.render_chain(base.data[bs:bs + bl], effects)
                        cache[key] = Source(data, base.name)
                    out.append(cache[key])
                progress((i + 1) / len(prepared))
            return out

        def done(results):
            st = self.state
            st.checkpoint()
            for (c, effects, _), src in zip(prepared, results):
                engine.apply_chain(c, effects, src)
            if after:
                after()
            st.changed.emit()
            self.statusBar().showMessage(f"{title}: готово", 4000)

        self.run_async(f"{title}…", work, done)

    def add_effect(self, kind, title, **params):
        clips = self._need_selection(title)
        if clips:
            fx = make_effect(kind, **params)
            self.process_clips([(c, c.effects + [fx]) for c in clips], title)

    def _need_selection(self, title):
        clips = self._selected_clips()
        if not clips:
            QMessageBox.information(self, title, "Выделите один или несколько клипов на таймлайне.")
        return clips

    def _effect_dialog(self, title, dialog_cls, kind):
        if not self._need_selection(title):
            return
        dlg = dialog_cls(self)
        if dlg.exec() == dialog_cls.Accepted:
            self.add_effect(kind, title, **dlg.values())

    def fx_volume(self):
        self._effect_dialog("Громкость", VolumeDialog, "volume")

    def fx_pitch(self):
        self._effect_dialog("Высота тона и темп", PitchDialog, "pitch")

    def fx_reverb(self):
        self._effect_dialog("Реверберация", ReverbDialog, "reverb")

    def fx_eq(self):
        self._effect_dialog("Эквалайзер", EQDialog, "eq")

    def fx_reverse(self):
        self.add_effect("reverse", "Реверс")

    def fx_channels(self, mode):
        self.add_effect("channels", CHANNEL_MODES[mode], mode=mode)

    def fx_noise(self):
        st = self.state
        clips = self._selected_clips()
        if not st.project.tracks:
            st.project.add_track()
            st.structure_changed.emit()
        dlg = NoiseDialog(self, len(clips), st.project.tracks[st.track].name)
        if dlg.exec() != NoiseDialog.Accepted:
            return
        v = dlg.values()
        fx = make_effect("noise", color=v["color"], level_db=v["level_db"])
        if clips:
            self.process_clips([(c, c.effects + [fx]) for c in clips], "Шум")
            return
        # без выделения: новый клип = тишина + эффект «Шум» (его можно потом убрать в менеджере)
        base = engine.silence_source(v["duration"] * SR)
        clip = Clip(base, st.track, st.cursor, 0, base.length, name="Шум")

        def after():
            st.project.clips.append(clip)
            st.selected = {clip}

        self.process_clips([(clip, [fx])], "Генерация шума", after)

    def insert_silence(self):
        st = self.state
        dur, ok = QInputDialog.getDouble(self, "Пустой клип", "Длительность тишины, секунд:",
                                         5.0, 0.05, 3600.0, 2)
        if not ok:
            return
        st.checkpoint()
        added = not st.project.tracks
        if added:
            st.project.add_track()
        clip = st.project.add_clip(engine.silence_source(dur * SR), st.track, st.cursor)
        st.selected = {clip}
        if added:
            st.structure_changed.emit()
        st.changed.emit()
        self.statusBar().showMessage("Создан пустой клип — к нему можно применить «Эффекты → Шум»", 5000)

    def effects_manager(self, clip=None):
        if clip is None:
            clips = self._need_selection("Активные эффекты")
            if not clips:
                return
            clip = clips[0]
        if not clip.effects:
            QMessageBox.information(self, "Активные эффекты", f"У клипа «{clip.name}» нет эффектов.")
            return
        dlg = EffectsManagerDialog(self, clip)
        if dlg.exec() != EffectsManagerDialog.Accepted:
            return
        new = dlg.values()
        if new != clip.effects:
            self.process_clips([(clip, new)], "Пересчёт эффектов")

    # ------------------------------------------------------------------ каналы
    def split_channels(self):
        """Каждый выделенный клип -> [Л] на своей дорожке и [П] на новой дорожке под ней."""
        clips = self._need_selection("Разделить каналы")
        if not clips:
            return
        st = self.state
        pairs = [(c, copy.copy(c)) for c in clips]
        jobs = []
        for left, right in pairs:
            jobs.append((left, left.effects + [make_effect("channels", mode="left")]))
            jobs.append((right, right.effects + [make_effect("channels", mode="right")]))

        def after():
            proj = st.project
            # вставляем снизу вверх, чтобы индексы верхних дорожек не съезжали
            for t in sorted({left.track for left, _ in pairs}, reverse=True):
                proj.insert_track(t + 1, f"{proj.tracks[t].name} — правый")
            for left, right in pairs:
                name = left.name
                left.name, right.name = f"{name} [Л]", f"{name} [П]"
                right.track = left.track + 1
                proj.clips.append(right)
            st.selected = {c for pair in pairs for c in pair}
            st.structure_changed.emit()

        self.process_clips(jobs, "Разделение каналов", after)

    def merge_channels(self):
        """Два выделенных клипа -> один стерео-клип: первый в левый канал, второй в правый."""
        st = self.state
        clips = self._selected_clips()
        if len(clips) != 2:
            QMessageBox.information(self, "Объединить каналы",
                                    "Выделите ровно два клипа (Ctrl+клик): будущий левый и правый канал.")
            return

        def mode(c):
            last = c.effects[-1] if c.effects else None
            return last["params"].get("mode") if last and last["kind"] == "channels" else None

        a, b = sorted(clips, key=lambda c: (c.track, c.start))
        if mode(a) == "right" or mode(b) == "left":
            a, b = b, a
        start = min(a.start, b.start)
        data = np.zeros((max(a.end, b.end) - start, 2), dtype=np.float32)
        for ch, c in ((0, a), (1, b)):
            mono = engine.clip_audio(c).mean(axis=1)
            data[c.start - start: c.start - start + len(mono), ch] = mono

        def strip(name):
            return name.replace(" [Л]", "").replace(" [П]", "")

        name = strip(a.name) if strip(a.name) == strip(b.name) else f"{strip(a.name)} + {strip(b.name)}"
        st.checkpoint()
        st.project.clips = [c for c in st.project.clips if c is not a and c is not b]
        merged = st.project.add_clip(Source(data, f"{name} (стерео)"), a.track, start)
        st.selected = {merged}
        st.changed.emit()
        self.statusBar().showMessage(f"Левый канал: «{a.name}», правый: «{b.name}»", 6000)

    # ------------------------------------------------------------------ вид
    def zoom(self, factor):
        st = self.state
        w = self.timeline.visible_width()
        x = st.x_of(st.cursor)
        st.set_zoom(st.pps * factor, x if 0 <= x <= w else w / 2)

    def zoom_fit(self):
        st = self.state
        end = st.project.end() / SR
        if end <= 0:
            return
        st.offset = 0.0
        st.set_zoom(self.timeline.visible_width() / (end * 1.05), 0)

    def track_height(self, delta):
        st = self.state
        st.track_h = min(max(st.track_h + delta, 60), 260)
        st.structure_changed.emit()

    def toggle_crossfade(self):
        self.state.project.auto_crossfade = self.a_xfade.isChecked()
        self.state.dirty = True
        self.state.changed.emit()

    def toggle_snap(self):
        self.state.snap = self.a_snap.isChecked()
        self.state.changed.emit()

    # ------------------------------------------------------------------ транспорт
    def play_stop(self):
        if self.player.playing:
            self.stop()
        else:
            self.start_playback()

    def pause(self):
        if self.player.playing:
            pos = self.player.position()
            self.stop()
            self.state.cursor = pos
            self.state.changed.emit()
        else:
            self.start_playback()

    def start_playback(self):
        st = self.state
        try:
            if self.a_loop.isChecked() and st.range:
                a, b = st.range
                start = st.cursor if a <= st.cursor < b else a
                self.player.play(start, b, loop=True, loop_start=a)
            else:
                end = st.project.end()
                if st.cursor >= end:
                    self.statusBar().showMessage("Курсор за концом проекта — нажмите Home", 3000)
                    return
                self.player.play(st.cursor, end)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Воспроизведение", str(e))
            self.player.stop()

    def _on_seek(self):
        if self.player.playing:          # курсор передвинули во время игры — играем оттуда
            self.start_playback()

    def stop(self):
        self.player.stop()
        self.state.playhead = None
        self.state.changed.emit()

    def go_start(self):
        st = self.state
        st.cursor = 0
        st.offset = 0.0
        st.changed.emit()

    def go_end(self):
        st = self.state
        st.cursor = st.project.end()
        w = self.timeline.visible_width() / st.pps
        if st.cursor / SR > st.offset + w:
            st.offset = max(0.0, st.cursor / SR - w * 0.8)
        st.changed.emit()

    def _tick(self):
        st = self.state
        if not self.player.playing:
            if self.meter.db.max() > -60:
                self.meter.set_levels([0, 0])
            return
        if not self.player.is_active():
            self.stop()
            return
        if self.player.backend == "ffplay":
            self.player.update_levels_fallback()
        old = st.playhead
        pos = self.player.position()
        st.playhead = pos
        self.meter.set_levels(self.player.levels)
        x = st.x_of(pos)
        w = self.timeline.visible_width()
        if x > w - 10 or x < 0:
            st.offset = max(0.0, pos / SR - w / st.pps * 0.05)
            st.changed.emit()
        else:
            self.timeline.update_playhead(old, pos)

    def show_help(self):
        QMessageBox.information(self, "Горячие клавиши", HELP_TEXT)

    def closeEvent(self, e):
        if not self.maybe_save():
            e.ignore()
            return
        self.player.stop()
        super().closeEvent(e)


HELP_TEXT = """\
Space — воспроизведение / стоп (курсор возвращается)
Enter — пауза (курсор остаётся на месте)
W (или Home) — в начало, End — в конец, Q — цикл по выделению

S — разрезать клипы под курсором (или выделенные)
Delete — удалить выделенные клипы (или диапазон, если клипы не выбраны)
Ctrl+Delete — удалить выделенный диапазон со сдвигом
Ctrl+X / C / V — вырезать / копировать / вставить в позицию курсора
Ctrl+Z / Ctrl+Y — отменить / повторить
Ctrl+A — выделить всё, Ctrl+T — новая дорожка
Ctrl+Shift+N — пустой клип (тишина), Ctrl+Shift+F — активные эффекты клипа

Мышь:
  тянуть курсор (пунктир или треугольник на линейке) — перемотка
  перетаскивание клипа — перемещение (в т.ч. между дорожками)
  Ctrl + перетаскивание клипа — копирование
  наложение клипов на одной дорожке — автоматический кроссфейд
  края клипа — обрезка; квадратики сверху — fade in / fade out
  протяжка по пустому месту или линейке — выделение диапазона
  Ctrl+клик — множественный выбор, двойной клик — свойства клипа
  колесо — прокрутка дорожек, Shift+колесо — по времени, Ctrl+колесо — масштаб
  файлы можно перетащить в окно мышью

Ctrl+O / Ctrl+S — открыть / сохранить проект
Ctrl+I — импорт, Ctrl+E — экспорт, N — привязка, Ctrl+F — показать всё
"""

CYRILLIC = dict(zip("QWERTYUIOPASDFGHJKLZXCVBNM", "ЙЦУКЕНГШЩЗФЫВАПРОЛДЯЧСМИТЬ"))

STYLE = """
QWidget { font-size: 9pt; }
QMainWindow, QToolBar { background: #2b2b2e; }
QToolBar { border-bottom: 1px solid #151515; spacing: 3px; }
#Transport { background: #252528; border-top: 1px solid #151515; }
#TimeBox { background: #050505; color: #3dff6e; font-size: 15pt; font-weight: bold;
           border-right: 1px solid #333; border-bottom: 1px solid #333; }
#HeaderColumn { background: #1b1b1c; }
#TrackHeader { background: #333337; border-bottom: 1px solid #141414; border-right: 1px solid #141414; }
#TrackHeader[selected="true"] { background: #3e4654; }
#TrackHeader QLineEdit { background: transparent; color: #eee; font-weight: bold; }
#TrackHeader QLineEdit:focus { background: #1d1d1d; }
#TrackHeader QToolButton { color: #ddd; }
QToolButton#trackBtnM:checked { background: #c9302c; color: white; }
QToolButton#trackBtnS:checked { background: #e0b030; color: black; }
QSlider::groove:horizontal { height: 4px; background: #1a1a1a; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; margin: -5px 0; background: #b8b8b8; border-radius: 3px; }
"""


def dark_palette():
    pal = QPalette()
    roles = {
        QPalette.Window: "#2b2b2e", QPalette.WindowText: "#e0e0e0", QPalette.Base: "#1e1e20",
        QPalette.AlternateBase: "#2b2b2e", QPalette.ToolTipBase: "#333", QPalette.ToolTipText: "#eee",
        QPalette.Text: "#e0e0e0", QPalette.Button: "#38383c", QPalette.ButtonText: "#e0e0e0",
        QPalette.Highlight: "#3f86d6", QPalette.HighlightedText: "#ffffff", QPalette.Link: "#5aa0f0",
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        pal.setColor(QPalette.Disabled, role, QColor("#777"))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Mini Vegas Audio")
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    files = [a for a in sys.argv[1:] if os.path.isfile(a)]
    projects = [f for f in files if f.lower().endswith("." + PROJECT_EXT)]
    media = [f for f in files if f not in projects]
    if projects:
        QTimer.singleShot(100, lambda: win.open_project(projects[0]))
    elif media:
        QTimer.singleShot(100, lambda: win.import_files(media, 0, 0))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
