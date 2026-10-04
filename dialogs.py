"""Диалоги: экспорт, эффекты, свойства клипа."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QRadioButton, QSlider, QSpinBox, QVBoxLayout)

from engine import (EQ_BANDS, EXPORT_FORMATS, NOISE_COLORS, SR, WAV_DEPTHS, describe_effect,
                    ffmpeg_has)

MP3_RATES = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}


def _spin(lo, hi, value, step=1.0, decimals=1, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


class _Base(QDialog):
    def __init__(self, parent, title, hint=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.lay = QVBoxLayout(self)
        if hint:
            lbl = QLabel(hint)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#aaa;")
            self.lay.addWidget(lbl)
        self.form = QFormLayout()
        self.lay.addLayout(self.form)

    def finish(self):
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Применить")
        bb.button(QDialogButtonBox.Cancel).setText("Отмена")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        self.lay.addWidget(bb)


# --------------------------------------------------------------------------- экспорт

class ExportDialog(QDialog):
    def __init__(self, parent, default_dir, has_range):
        super().__init__(parent)
        self.setWindowTitle("Экспорт (Render As)")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)

        self.fmt = QComboBox()
        for key, (label, _, _) in EXPORT_FORMATS.items():
            self.fmt.addItem(label, key)
        form.addRow("Формат:", self.fmt)

        self.rate = QComboBox()
        self.rate.setEditable(True)
        for r in (8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000):
            self.rate.addItem(str(r))
        self.rate.setCurrentText("44100")
        self.rate.setValidator(QIntValidator(1000, 768000, self))
        form.addRow("Частота дискретизации (Гц):", self.rate)

        self.bitrate = QComboBox()
        self.bitrate.setEditable(True)
        for b in (32, 48, 64, 96, 128, 160, 192, 224, 256, 320):
            self.bitrate.addItem(str(b))
        self.bitrate.setCurrentText("192")
        self.bitrate.setValidator(QIntValidator(8, 1024, self))
        form.addRow("Битрейт (кбит/с):", self.bitrate)

        self.depth = QComboBox()
        self.depth.addItems(list(WAV_DEPTHS))
        form.addRow("Разрядность WAV:", self.depth)

        self.channels = QComboBox()
        self.channels.addItem("Стерео", 2)
        self.channels.addItem("Моно", 1)
        form.addRow("Каналы:", self.channels)

        self.all_rb = QRadioButton("Весь проект")
        self.range_rb = QRadioButton("Только выделенный диапазон")
        self.all_rb.setChecked(True)
        self.range_rb.setEnabled(has_range)
        if has_range:
            self.range_rb.setChecked(True)
        rbox = QHBoxLayout()
        rbox.addWidget(self.all_rb)
        rbox.addWidget(self.range_rb)
        form.addRow("Что экспортировать:", rbox)

        self.path = QLineEdit(os.path.join(default_dir, "render.mp3"))
        browse = QPushButton("Обзор…")
        browse.clicked.connect(self._browse)
        prow = QHBoxLayout()
        prow.addWidget(self.path)
        prow.addWidget(browse)
        form.addRow("Файл:", prow)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#d0a040;")
        lay.addWidget(self.hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Экспорт")
        bb.button(QDialogButtonBox.Cancel).setText("Отмена")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        self.rate.currentTextChanged.connect(self._update_hint)
        self._fmt_changed()

    def _fmt_changed(self):
        key = self.fmt.currentData()
        _, ext, has_br = EXPORT_FORMATS[key]
        self.bitrate.setEnabled(has_br)
        self.depth.setEnabled(key == "wav")
        root, _ = os.path.splitext(self.path.text())
        self.path.setText(f"{root}.{ext}")
        self._update_hint()

    def _update_hint(self):
        key = self.fmt.currentData()
        try:
            rate = int(self.rate.currentText())
        except ValueError:
            rate = 0
        msg = ""
        if key == "mp3" and rate not in MP3_RATES:
            msg = "MP3 поддерживает только 8000–48000 Гц (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)."
        elif key == "mp4" and rate > 96000:
            msg = "AAC обычно поддерживает частоты до 96000 Гц."
        elif key == "ogg" and not ffmpeg_has("encoders", "libvorbis"):
            msg = "libvorbis не найден — будет использован встроенный экспериментальный кодер vorbis."
        self.hint.setText(msg)

    def _browse(self):
        key = self.fmt.currentData()
        _, ext, _ = EXPORT_FORMATS[key]
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить как", self.path.text(),
                                              f"{ext.upper()} (*.{ext})")
        if path:
            if not path.lower().endswith("." + ext):
                path += "." + ext
            self.path.setText(path)

    def _accept(self):
        try:
            int(self.rate.currentText())
            int(self.bitrate.currentText())
        except ValueError:
            QMessageBox.warning(self, "Экспорт", "Укажите частоту и битрейт числами.")
            return
        if not self.path.text().strip():
            QMessageBox.warning(self, "Экспорт", "Укажите имя файла.")
            return
        self.accept()

    def values(self):
        key = self.fmt.currentData()
        ext = EXPORT_FORMATS[key][1]
        path = os.path.expanduser(self.path.text().strip())
        if not path.lower().endswith("." + ext):
            path += "." + ext
        return dict(path=path, fmt=key, sample_rate=int(self.rate.currentText()),
                    channels=self.channels.currentData(), bitrate_k=int(self.bitrate.currentText()),
                    wav_codec=WAV_DEPTHS[self.depth.currentText()],
                    use_range=self.range_rb.isChecked())


# --------------------------------------------------------------------------- эффекты

class VolumeDialog(_Base):
    def __init__(self, parent):
        super().__init__(parent, "Громкость", "Изменение уровня выделенных клипов (фильтр ffmpeg volume).")
        self.gain = _spin(-60, 30, 0, 0.5, 1, "дБ")
        self.normalize = QCheckBox("Нормализовать по пику")
        self.target = _spin(-30, 0, -1, 0.5, 1, "дБFS")
        self.target.setEnabled(False)
        self.normalize.toggled.connect(self.target.setEnabled)
        self.normalize.toggled.connect(lambda on: self.gain.setEnabled(not on))
        self.form.addRow("Усиление:", self.gain)
        self.form.addRow(self.normalize)
        self.form.addRow("Целевой пик:", self.target)
        self.finish()

    def values(self):
        return dict(gain_db=self.gain.value(), normalize=self.normalize.isChecked(),
                    target_db=self.target.value())


class PitchDialog(_Base):
    def __init__(self, parent):
        rb = ffmpeg_has("filters", "rubberband")
        engine = "rubberband (высокое качество)" if rb else "asetrate + atempo"
        super().__init__(parent, "Высота тона и темп", f"Алгоритм сдвига тона: {engine}.")
        self.semi = _spin(-24, 24, 0, 1, 2, "полутонов")
        self.tempo = _spin(25, 400, 100, 5, 1, "%")
        self.keep = QCheckBox("Сохранять длительность при смене тона")
        self.keep.setChecked(True)
        self.keep.setToolTip("Если выключено — тон и скорость меняются вместе, как у пластинки.")
        self.form.addRow("Сдвиг тона:", self.semi)
        self.form.addRow("Темп (скорость):", self.tempo)
        self.form.addRow(self.keep)
        self.finish()

    def values(self):
        return dict(semitones=self.semi.value(), tempo_pct=self.tempo.value(),
                    keep_duration=self.keep.isChecked())


REVERB_PRESETS = {
    "Маленькая комната": (0.5, 5, 0.6, 22, 100, 70),
    "Средняя комната": (1.0, 12, 0.5, 28, 100, 85),
    "Концертный зал": (2.4, 25, 0.4, 32, 100, 100),
    "Собор": (5.5, 45, 0.3, 40, 90, 100),
    "Пластина": (1.6, 0, 0.15, 30, 100, 100),
}


class ReverbDialog(_Base):
    def __init__(self, parent):
        super().__init__(parent, "Реверберация",
                         "Свёрточная реверберация с синтетической импульсной характеристикой. "
                         "Клип удлиняется на «хвост» реверберации.")
        self.preset = QComboBox()
        self.preset.addItems(list(REVERB_PRESETS))
        self.decay = _spin(0.1, 12, 2.4, 0.1, 2, "с")
        self.predelay = _spin(0, 300, 25, 1, 0, "мс")
        self.damping = QSpinBox()
        self.damping.setRange(0, 100)
        self.damping.setSuffix(" %")
        self.wet = QSpinBox()
        self.wet.setRange(0, 100)
        self.wet.setSuffix(" %")
        self.dry = QSpinBox()
        self.dry.setRange(0, 100)
        self.dry.setSuffix(" %")
        self.width = QSpinBox()
        self.width.setRange(0, 100)
        self.width.setSuffix(" %")
        self.form.addRow("Пресет:", self.preset)
        self.form.addRow("Время затухания (RT60):", self.decay)
        self.form.addRow("Предзадержка:", self.predelay)
        self.form.addRow("Демпфирование ВЧ:", self.damping)
        self.form.addRow("Обработанный сигнал (wet):", self.wet)
        self.form.addRow("Исходный сигнал (dry):", self.dry)
        self.form.addRow("Стереоширина:", self.width)
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.preset.setCurrentText("Концертный зал")
        self._apply_preset("Концертный зал")
        self.finish()

    def _apply_preset(self, name):
        decay, pre, damp, wet, dry, width = REVERB_PRESETS[name]
        self.decay.setValue(decay)
        self.predelay.setValue(pre)
        self.damping.setValue(int(damp * 100))
        self.wet.setValue(wet)
        self.dry.setValue(dry)
        self.width.setValue(width)

    def values(self):
        return dict(decay_s=self.decay.value(), predelay_ms=self.predelay.value(),
                    damping=self.damping.value() / 100, wet=self.wet.value() / 100,
                    dry=self.dry.value() / 100, width=self.width.value() / 100)


EQ_PRESETS = {
    "Ровно": [0] * 10,
    "Подъём басов": [6, 5, 4, 2, 0, 0, 0, 0, 0, 0],
    "Подъём верхов": [0, 0, 0, 0, 0, 1, 2, 4, 5, 6],
    "Голос": [-6, -4, -2, 0, 2, 3, 3, 2, 0, -2],
    "Громкость (loudness)": [6, 4, 1, 0, -1, 0, 0, 1, 4, 6],
    "Телефон": [-12, -12, -8, 0, 4, 6, 4, 0, -12, -12],
    "Убрать гул": [-12, -8, -2, 0, 0, 0, 0, 0, 0, 0],
}


class EQDialog(_Base):
    def __init__(self, parent):
        super().__init__(parent, "Эквалайзер", "10-полосный графический эквалайзер (фильтр ffmpeg equalizer).")
        self.preset = QComboBox()
        self.preset.addItems(list(EQ_PRESETS))
        self.form.addRow("Пресет:", self.preset)
        grid = QGridLayout()
        self.sliders = []
        self.labels = []
        for i, f in enumerate(EQ_BANDS):
            s = QSlider(Qt.Vertical)
            s.setRange(-120, 120)
            s.setTickPosition(QSlider.TicksBothSides)
            s.setTickInterval(30)
            s.setMinimumHeight(170)
            val = QLabel("0.0")
            val.setAlignment(Qt.AlignCenter)
            s.valueChanged.connect(lambda v, lbl=val: lbl.setText(f"{v / 10:+.1f}"))
            name = QLabel(f"{f // 1000}k" if f >= 1000 else str(f))
            name.setAlignment(Qt.AlignCenter)
            grid.addWidget(val, 0, i)
            grid.addWidget(s, 1, i, Qt.AlignHCenter)
            grid.addWidget(name, 2, i)
            self.sliders.append(s)
        box = QGroupBox("Полосы, дБ (±12)")
        box.setLayout(grid)
        self.lay.addWidget(box)
        filt = QFormLayout()
        self.hp_on = QCheckBox("Срез низов (HPF)")
        self.hp = _spin(10, 2000, 80, 10, 0, "Гц")
        self.lp_on = QCheckBox("Срез верхов (LPF)")
        self.lp = _spin(500, 20000, 12000, 100, 0, "Гц")
        filt.addRow(self.hp_on, self.hp)
        filt.addRow(self.lp_on, self.lp)
        self.lay.addLayout(filt)
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.finish()

    def _apply_preset(self, name):
        for s, g in zip(self.sliders, EQ_PRESETS[name]):
            s.setValue(int(g * 10))

    def values(self):
        return dict(gains=[s.value() / 10 for s in self.sliders],
                    highpass=self.hp.value() if self.hp_on.isChecked() else None,
                    lowpass=self.lp.value() if self.lp_on.isChecked() else None)


class NoiseDialog(_Base):
    def __init__(self, parent, n_selected, track_name):
        if n_selected:
            hint = f"Шум будет наложен на выделенные клипы ({n_selected} шт.)."
        else:
            hint = f"Клипы не выделены — будет создан новый клип с шумом на дорожке «{track_name}» в позиции курсора."
        super().__init__(parent, "Шум", hint + " Генерация через ffmpeg anoisesrc.")
        self.color = QComboBox()
        self.color.addItems(list(NOISE_COLORS))
        self.level = _spin(-80, 0, -30, 1, 1, "дБFS (RMS)")
        self.duration = _spin(0.1, 3600, 5, 1, 1, "с")
        self.duration.setEnabled(not n_selected)
        self.form.addRow("Цвет шума:", self.color)
        self.form.addRow("Уровень:", self.level)
        self.form.addRow("Длительность (новый клип):", self.duration)
        self.finish()

    def values(self):
        return dict(color=NOISE_COLORS[self.color.currentText()], level_db=self.level.value(),
                    duration=self.duration.value())


class ClipPropertiesDialog(_Base):
    def __init__(self, parent, clip):
        src = clip.source
        info = f"Источник: {src.name}  ({src.length / SR:.2f} с)"
        if src.path:
            info += f"\n{src.path}"
        super().__init__(parent, "Свойства клипа", info)
        self.name = QLineEdit(clip.name)
        self.gain = _spin(-60, 24, clip.gain_db, 0.5, 1, "дБ")
        self.fade_in = _spin(0, clip.length / SR * 1000, clip.fade_in / SR * 1000, 10, 0, "мс")
        self.fade_out = _spin(0, clip.length / SR * 1000, clip.fade_out / SR * 1000, 10, 0, "мс")
        self.form.addRow("Название:", self.name)
        self.form.addRow("Громкость клипа:", self.gain)
        self.form.addRow("Нарастание (fade in):", self.fade_in)
        self.form.addRow("Затухание (fade out):", self.fade_out)
        self.form.addRow("Длительность:", QLabel(f"{clip.length / SR:.3f} с"))
        self.finish()

    def values(self):
        return dict(name=self.name.text(), gain_db=self.gain.value(),
                    fade_in=int(self.fade_in.value() / 1000 * SR),
                    fade_out=int(self.fade_out.value() / 1000 * SR))


class EffectsManagerDialog(QDialog):
    """Активные эффекты клипа: включить/выключить, поменять порядок, удалить."""

    def __init__(self, parent, clip):
        super().__init__(parent)
        self.setWindowTitle(f"Эффекты клипа «{clip.name}»")
        self.setMinimumSize(460, 320)
        lay = QVBoxLayout(self)
        hint = QLabel("Эффекты применяются сверху вниз к исходному звуку клипа. "
                      "Снимите галочку, чтобы временно отключить эффект.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa;")
        lay.addWidget(hint)
        row = QHBoxLayout()
        self.list = QListWidget()
        row.addWidget(self.list, 1)
        buttons = QVBoxLayout()
        for text, slot in (("Вверх", lambda: self._move(-1)), ("Вниз", lambda: self._move(1)),
                           ("Удалить", self._remove), ("Удалить все", self.list.clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        row.addLayout(buttons)
        lay.addLayout(row)
        for fx in clip.effects:
            self._add_item(fx)
        if self.list.count():
            self.list.setCurrentRow(0)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Применить")
        bb.button(QDialogButtonBox.Cancel).setText("Отмена")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _add_item(self, fx, row=None):
        item = QListWidgetItem(describe_effect(fx))
        item.setData(Qt.UserRole, fx)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if fx.get("enabled", True) else Qt.Unchecked)
        if row is None:
            self.list.addItem(item)
        else:
            self.list.insertItem(row, item)
        return item

    def _move(self, delta):
        row = self.list.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < self.list.count()):
            return
        item = self.list.takeItem(row)
        self.list.insertItem(new, item)
        self.list.setCurrentRow(new)

    def _remove(self):
        row = self.list.currentRow()
        if row >= 0:
            self.list.takeItem(row)

    def values(self):
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            fx = dict(item.data(Qt.UserRole))
            fx["enabled"] = item.checkState() == Qt.Checked
            out.append(fx)
        return out
