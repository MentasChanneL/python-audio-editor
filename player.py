"""Воспроизведение: sounddevice (живое сведение) или ffplay как запасной вариант."""
import os
import shutil
import subprocess
import tempfile
import time
import wave

import numpy as np

from engine import SR

try:
    import sounddevice as sd
    _SD_ERROR = None
except Exception as e:  # нет модуля или libportaudio2
    sd = None
    _SD_ERROR = e


class Player:
    def __init__(self, project_getter):
        self._project = project_getter
        self._stream = None
        self._proc = None
        self._tmp = None
        self.playing = False
        self.levels = np.zeros(2)
        self.pos = 0
        self.start = 0
        self.end = 0
        self.loop = False
        self.loop_start = 0
        if sd is not None:
            self.backend = "sounddevice"
        elif shutil.which("ffplay"):
            self.backend = "ffplay"
        else:
            self.backend = None

    def backend_error(self):
        return ("Нет аудиовывода. Установите sounddevice (pip install sounddevice; "
                "sudo apt install libportaudio2) или ffplay (входит в пакет ffmpeg).\n"
                f"Подробности: {_SD_ERROR}")

    # ------------------------------------------------------------------ управление
    def play(self, start, end, loop=False, loop_start=0):
        self.stop()
        if self.backend is None:
            raise RuntimeError(self.backend_error())
        self.pos = self.start = int(start)
        self.end = int(end)
        self.loop = loop
        self.loop_start = int(loop_start)
        self._wrapped = False
        self.levels = np.zeros(2)
        if self.backend == "sounddevice":
            self._stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32",
                                           blocksize=1024, latency="high", callback=self._callback)
            self._stream.start()
        else:
            self._start_ffplay()
        self.playing = True

    def stop(self):
        self.playing = False
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._tmp:
            try:
                os.remove(self._tmp)
            except OSError:
                pass
            self._tmp = None
        self.levels = np.zeros(2)

    def is_active(self):
        if self._stream is not None:
            return self._stream.active
        if self._proc is not None:
            return self._proc.poll() is None
        return False

    def position(self):
        """Позиция на таймлайне (сэмплы), с учётом задержки вывода."""
        if self._stream is not None:
            p = self.pos - int(self._stream.latency * SR)
            if self._wrapped and p < self.loop_start:
                p = self.end - (self.loop_start - p)   # звучит ещё конец предыдущего круга
            return max(p, self.start if not self._wrapped else self.loop_start)
        if self._proc is not None:
            elapsed = int((time.monotonic() - self._t0) * SR)
            if self.loop:
                first = self.end - self.start
                if elapsed < first:
                    return self.start + elapsed
                span = max(self.end - self.loop_start, 1)
                return self.loop_start + (elapsed - first) % span
            return min(self.start + elapsed, self.end)
        return self.pos

    # ------------------------------------------------------------------ sounddevice
    def _callback(self, outdata, frames, time_info, status):
        finished = False
        try:
            project = self._project()
            filled = 0
            while filled < frames:
                if self.pos >= self.end:
                    if self.loop and self.end > self.loop_start:
                        self.pos = self.loop_start
                        self._wrapped = True
                    else:
                        outdata[filled:] = 0
                        finished = True
                        break
                k = min(frames - filled, self.end - self.pos)
                outdata[filled:filled + k] = project.render(self.pos, k)
                self.pos += k
                filled += k
        except Exception:
            outdata[:] = 0
        self.levels = np.abs(outdata).max(axis=0)
        if finished:
            raise sd.CallbackStop

    # ------------------------------------------------------------------ ffplay
    def _start_ffplay(self):
        # ffplay умеет зацикливать только весь файл, поэтому для цикла рендерим сам цикл,
        # а предварительный «разгон» от курсора не поддерживаем — начинаем с начала цикла.
        if self.loop:
            self.start = self.loop_start
        buf = self._project().render(self.start, self.end - self.start)
        self._buf = buf
        fd, self._tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(self._tmp, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((buf * 32767).astype("<i2").tobytes())
        cmd = [shutil.which("ffplay"), "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if self.loop:
            cmd += ["-loop", "0"]
        self._proc = subprocess.Popen(cmd + [self._tmp], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._t0 = time.monotonic()

    def update_levels_fallback(self):
        if self._proc is None or not hasattr(self, "_buf"):
            return
        i = self.position() - self.start
        win = self._buf[max(i - 1024, 0): i + 1024]
        self.levels = np.abs(win).max(axis=0) if len(win) else np.zeros(2)
