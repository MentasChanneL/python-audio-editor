"""Ядро редактора: ввод/вывод через ffmpeg, модель проекта, микшер и эффекты."""
import copy
import functools
import itertools
import os
import shutil
import subprocess
import json
import tempfile
from dataclasses import dataclass, field

import numpy as np

SR = 44100          # внутренняя частота дискретизации проекта
PEAK_BLOCK = 256    # размер блока для первого уровня пиков
PEAK_FACTOR = 16    # во сколько раз огрубляется каждый следующий уровень

IMPORT_EXTS = ("mp3", "ogg", "oga", "wav", "mp4", "m4a", "flac", "aac", "opus",
               "webm", "mkv", "mov", "avi", "wma")


class FFmpegError(RuntimeError):
    pass


def db_to_lin(db):
    return 10.0 ** (db / 20.0)


def lin_to_db(x):
    return 20.0 * np.log10(max(float(x), 1e-9))


# --------------------------------------------------------------------------- ffmpeg

def ffmpeg_bin():
    path = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not path:
        raise FFmpegError("ffmpeg не найден в PATH.\nУстановите его: sudo apt install ffmpeg")
    return path


def _run_ffmpeg(args, input_bytes=None):
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-v", "error", *args]
    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise FFmpegError(msg[-3000:] or f"ffmpeg завершился с кодом {proc.returncode}")
    return proc.stdout


@functools.lru_cache(maxsize=None)
def _ffmpeg_list(kind):
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", f"-{kind}"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except (OSError, FFmpegError):
        return frozenset()
    names = set()
    for line in out.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return frozenset(names)


def ffmpeg_has(kind, name):
    """kind: 'filters' или 'encoders'."""
    return name in _ffmpeg_list(kind)


def _to_frames(raw, channels=2):
    data = np.frombuffer(raw, dtype=np.float32)
    data = data[: len(data) - len(data) % channels]
    return data.reshape(-1, channels)


def decode_file(path):
    """Декодирует любой файл, который понимает ffmpeg (из видео берётся только звук)."""
    try:
        raw = _run_ffmpeg(["-i", path, "-map", "0:a:0", "-vn", "-sn", "-dn",
                           "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"])
    except FFmpegError as e:
        if "matches no streams" in str(e):
            raise FFmpegError(f"В файле нет аудиодорожки:\n{path}") from None
        raise
    data = _to_frames(raw)
    if len(data) == 0:
        raise FFmpegError(f"Не удалось получить звук из файла:\n{path}")
    return data


def filter_audio(data, af):
    """Пропускает стерео-сигнал через цепочку аудиофильтров ffmpeg."""
    if not af:
        return data
    raw = _run_ffmpeg(["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                       "-af", af, "-f", "f32le", "-ar", str(SR), "-ac", "2", "-"],
                      input_bytes=np.ascontiguousarray(data, dtype=np.float32).tobytes())
    out = _to_frames(raw)
    if len(out) == 0:
        raise FFmpegError("Фильтр вернул пустой результат")
    return out


def generate_noise(color, n_samples):
    """Стерео-шум заданного цвета через источник anoisesrc (каналы с разным зерном)."""
    dur = n_samples / SR + 0.1
    chans = []
    for seed in (1234, 5678):
        raw = _run_ffmpeg(["-f", "lavfi", "-i",
                           f"anoisesrc=d={dur:.3f}:c={color}:r={SR}:a=0.5:s={seed}",
                           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"])
        ch = np.frombuffer(raw, dtype=np.float32)
        if len(ch) < n_samples:
            ch = np.pad(ch, (0, n_samples - len(ch)))
        chans.append(ch[:n_samples])
    return np.stack(chans, axis=1)


# --------------------------------------------------------------------------- экспорт

EXPORT_FORMATS = {
    # ключ: (подпись, расширение, есть ли битрейт)
    "mp3": ("MP3 (LAME)", "mp3", True),
    "ogg": ("OGG Vorbis", "ogg", True),
    "wav": ("WAV (PCM)", "wav", False),
    "mp4": ("MP4 (AAC)", "mp4", True),
    "flac": ("FLAC (без потерь)", "flac", False),
}

WAV_DEPTHS = {"16 бит": "pcm_s16le", "24 бит": "pcm_s24le", "32 бит float": "pcm_f32le"}


def _codec_args(fmt, bitrate_k, wav_codec):
    if fmt == "mp3":
        args = ["-c:a", "libmp3lame"]
    elif fmt == "ogg":
        if ffmpeg_has("encoders", "libvorbis"):
            args = ["-c:a", "libvorbis"]
        else:
            args = ["-c:a", "vorbis", "-strict", "-2"]
    elif fmt == "mp4":
        args = ["-c:a", "aac", "-movflags", "+faststart"]
    elif fmt == "flac":
        return ["-c:a", "flac"]
    else:
        return ["-c:a", wav_codec]
    return args + ["-b:a", f"{int(bitrate_k)}k"]


def _vorbis_quality(bitrate_k, channels):
    """Примерное соответствие битрейта шкале качества Vorbis (-q:a), для стерео 44.1 кГц."""
    per_stereo = bitrate_k * (2 / channels)
    table = [(64, 0), (80, 1), (96, 2), (112, 3), (128, 4), (160, 5), (192, 6), (224, 7), (256, 8), (320, 9)]
    return min(table, key=lambda t: abs(t[0] - per_stereo))[1] if per_stereo <= 400 else 10


def export_project(project, path, fmt, sample_rate, channels, bitrate_k=192,
                   wav_codec="pcm_s16le", start=0, end=None, progress=None):
    """Рендерит проект кусками и отправляет в ffmpeg. progress(frac) -> False для отмены."""
    if end is None:
        end = project.end()
    if end <= start:
        raise FFmpegError("Нечего экспортировать: проект пуст")
    codec = _codec_args(fmt, bitrate_k, wav_codec)
    try:
        return _export_once(project, path, sample_rate, channels, codec, start, end, progress)
    except FFmpegError as e:
        # libvorbis принимает не любой битрейт для данной частоты/числа каналов —
        # тогда кодируем в режиме качества с ближайшим средним битрейтом.
        if fmt != "ogg" or "encoder setup failed" not in str(e):
            raise
        codec = codec[:codec.index("-b:a")] + ["-q:a", str(_vorbis_quality(bitrate_k, channels))]
        return _export_once(project, path, sample_rate, channels, codec, start, end, progress)


def _export_once(project, path, sample_rate, channels, codec, start, end, progress):
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-v", "error", "-y",
           "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-vn",
           "-ar", str(int(sample_rate)), "-ac", str(int(channels)), *codec, path]
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf)
        cancelled = False
        try:
            chunk = SR * 2
            for pos in range(start, end, chunk):
                buf = project.render(pos, min(chunk, end - pos))
                proc.stdin.write(buf.tobytes())
                if progress and progress((pos - start) / (end - start)) is False:
                    cancelled = True
                    break
        except OSError:  # ffmpeg упал и закрыл канал — причина будет в stderr
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if cancelled:
            proc.kill()
            proc.wait()
            try:
                os.remove(path)
            except OSError:
                pass
            return False
        rc = proc.wait()
        if rc != 0:
            errf.seek(0)
            msg = errf.read().decode("utf-8", "replace").strip()
            raise FFmpegError(msg[-3000:] or f"ffmpeg завершился с кодом {rc}")
    return True


# --------------------------------------------------------------------------- пики

def build_peaks(data):
    """Пирамида пиков: список (размер_блока, min, max, mean_square), каждый массив (n, 2)."""
    levels = []
    n = len(data) // PEAK_BLOCK
    if n == 0:
        return levels
    d = data[: n * PEAK_BLOCK].reshape(n, PEAK_BLOCK, 2)
    mn, mx = d.min(axis=1), d.max(axis=1)
    sq = np.einsum("ijk,ijk->ik", d, d) / PEAK_BLOCK
    block = PEAK_BLOCK
    levels.append((block, mn, mx, sq))
    while len(mn) >= PEAK_FACTOR * 8:
        k = len(mn) // PEAK_FACTOR
        mn = mn[: k * PEAK_FACTOR].reshape(k, PEAK_FACTOR, 2).min(axis=1)
        mx = mx[: k * PEAK_FACTOR].reshape(k, PEAK_FACTOR, 2).max(axis=1)
        sq = sq[: k * PEAK_FACTOR].reshape(k, PEAK_FACTOR, 2).mean(axis=1)
        block *= PEAK_FACTOR
        levels.append((block, mn, mx, sq))
    return levels


def column_peaks(source, starts, end):
    """min/max/rms для колонок пикселей. starts — возрастающие индексы сэмплов источника."""
    spp = (end - starts[0]) / max(len(starts), 1)
    block = 1
    for lv in source.peaks:
        if lv[0] <= spp:
            block, mn_a, mx_a, sq_a = lv
    total = source.length if block == 1 else len(mn_a)
    base = min(max(int(starts[0]) // block, 0), total - 1)
    stop = min(max(-(-int(end) // block), base + 1), total)
    idx = np.clip(starts // block - base, 0, stop - base - 1).astype(np.int64)
    if block == 1:
        seg = source.data[base:stop]
        mn_s, mx_s, sq_s = seg, seg, seg * seg
    else:
        mn_s, mx_s, sq_s = mn_a[base:stop], mx_a[base:stop], sq_a[base:stop]
    mn = np.minimum.reduceat(mn_s, idx, axis=0)
    mx = np.maximum.reduceat(mx_s, idx, axis=0)
    sqs = np.add.reduceat(sq_s, idx, axis=0)
    counts = np.diff(np.append(idx, stop - base)).clip(min=1)[:, None]
    return mn, mx, np.sqrt(sqs / counts)


# --------------------------------------------------------------------------- модель

class Source:
    """Неизменяемый кусок декодированного звука (N, 2) float32.

    origin описывает, как воссоздать звук при открытии проекта:
    {"type": "file", "path": ...}, {"type": "silence", "n": ...} или None
    (тогда при сохранении звук пишется в папку media рядом с проектом).
    """
    _ids = itertools.count(1)

    def __init__(self, data, name, path=None, origin=None):
        self.id = next(Source._ids)
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.name = name
        self.path = path
        self.origin = origin if origin is not None else ({"type": "file", "path": path} if path else None)
        self.length = len(self.data)
        self.peaks = build_peaks(self.data)


def silence_source(n_samples, name="Тишина"):
    return Source(np.zeros((int(n_samples), 2), dtype=np.float32), name,
                  origin={"type": "silence", "n": int(n_samples)})


@dataclass
class Track:
    name: str
    volume_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    color: int = 0


@dataclass(eq=False)
class Clip:
    source: Source      # то, что звучит (после цепочки эффектов)
    track: int
    start: int          # позиция на таймлайне, сэмплы
    src_start: int      # смещение внутри source
    length: int
    gain_db: float = 0.0
    fade_in: int = 0
    fade_out: int = 0
    name: str = ""
    # Неразрушающая цепочка эффектов: source = render_chain(fx_base[fx_base_start:+fx_base_len]).
    # Список и словари внутри никогда не меняются на месте — только заменяются целиком,
    # потому что снимки для отмены и копии клипов разделяют их.
    fx_base: Source = None
    fx_base_start: int = 0
    fx_base_len: int = 0
    effects: list = field(default_factory=list)

    @property
    def end(self):
        return self.start + self.length

    def base(self):
        """(исходный источник, начало, длина) — то, к чему применяется цепочка."""
        if self.fx_base is not None:
            return self.fx_base, self.fx_base_start, self.fx_base_len
        return self.source, self.src_start, self.length

    def clamp_fades(self):
        self.fade_in = max(0, min(self.fade_in, self.length))
        self.fade_out = max(0, min(self.fade_out, self.length - self.fade_in))


MIN_CLIP = 64


def effects_key(base, start, length, effects):
    return (base.id, start, length, json.dumps(effects, sort_keys=True))


def apply_chain(clip, effects, processed):
    """Ставит клипу новую цепочку эффектов и её результат processed (Source или None,
    если цепочка пуста), сохраняя по возможности обрезку клипа."""
    base, bs, bl = clip.base()
    had = clip.fx_base is not None
    covered = (not had) or (clip.src_start == 0 and clip.length == clip.source.length)
    off = clip.src_start if had else 0
    if not effects:
        clip.source, clip.fx_base, clip.effects = base, None, []
        if covered:
            clip.src_start, clip.length = bs, bl
        else:
            clip.src_start = max(0, min(bs + off, base.length - MIN_CLIP))
            clip.length = max(1, min(clip.length, base.length - clip.src_start))
    else:
        clip.fx_base, clip.fx_base_start, clip.fx_base_len = base, bs, bl
        clip.effects = list(effects)
        clip.source = processed
        if covered:
            clip.src_start, clip.length = 0, processed.length
        else:
            clip.src_start = max(0, min(off, processed.length - MIN_CLIP))
            clip.length = max(1, min(clip.length, processed.length - clip.src_start))
    clip.clamp_fades()


def clip_envelope(clip, pos, auto=(0, 0)):
    """Огибающая громкости клипа в позициях pos (сэмплы от начала клипа).
    Ручные фейды линейные, автоматические кроссфейды — равномощные (sin)."""
    env = np.ones(len(pos), dtype=np.float32)
    ai, ao = auto
    if ai > clip.fade_in:
        env *= np.sin(np.clip(pos / ai, 0.0, 1.0) * (np.pi / 2))
    elif clip.fade_in > 0:
        env *= np.clip(pos / clip.fade_in, 0.0, 1.0)
    if ao > clip.fade_out:
        env *= np.sin(np.clip((clip.length - pos) / ao, 0.0, 1.0) * (np.pi / 2))
    elif clip.fade_out > 0:
        env *= np.clip((clip.length - pos) / clip.fade_out, 0.0, 1.0)
    return env


def clip_audio(clip):
    """Звук клипа с учётом его громкости и ручных фейдов (без настроек дорожки)."""
    seg = clip.source.data[clip.src_start:clip.src_start + clip.length]
    env = clip_envelope(clip, np.arange(len(seg), dtype=np.float32)) * db_to_lin(clip.gain_db)
    return seg * env[:, None]


class Project:
    def __init__(self):
        self.tracks = []
        self.clips = []
        self.auto_crossfade = True
        self._color_counter = 0

    def end(self):
        return max((c.end for c in self.clips), default=0)

    def snapshot(self):
        return ([copy.copy(t) for t in self.tracks], [copy.copy(c) for c in self.clips])

    def restore(self, snap):
        tracks, clips = snap
        self.tracks = [copy.copy(t) for t in tracks]
        self.clips = [copy.copy(c) for c in clips]

    def add_track(self, name=None):
        return self.insert_track(len(self.tracks), name)

    def insert_track(self, index, name=None):
        self.tracks.insert(index, Track(name or f"Дорожка {len(self.tracks) + 1}",
                                        color=self._color_counter))
        self._color_counter += 1
        for c in self.clips:
            if c.track >= index:
                c.track += 1
        return index

    def remove_track(self, index):
        self.clips = [c for c in self.clips if c.track != index]
        for c in self.clips:
            if c.track > index:
                c.track -= 1
        del self.tracks[index]

    def add_clip(self, source, track, start):
        clip = Clip(source, track, int(start), 0, source.length, name=source.name)
        self.clips.append(clip)
        return clip

    def split_clip(self, clip, pos):
        """Режет клип в позиции pos (таймлайн). Возвращает правую часть или None."""
        if not (clip.start + MIN_CLIP <= pos <= clip.end - MIN_CLIP):
            return None
        left_len = pos - clip.start
        right = copy.copy(clip)
        right.start = pos
        right.src_start = clip.src_start + left_len
        right.length = clip.length - left_len
        right.fade_in = 0
        clip.length = left_len
        clip.fade_out = 0
        clip.clamp_fades()
        right.clamp_fades()
        self.clips.insert(self.clips.index(clip) + 1, right)
        return right

    def delete_range(self, a, b):
        """Удаляет промежуток [a, b) на всех дорожках и сдвигает всё, что правее, влево."""
        for c in list(self.clips):
            if c.start < a < c.end:
                self.split_clip(c, a)
        for c in list(self.clips):
            if c.start < b < c.end:
                self.split_clip(c, b)
        width = b - a
        kept = []
        for c in self.clips:
            if c.start >= a and c.end <= b:
                continue
            if c.start >= b:
                c.start -= width
            kept.append(c)
        self.clips = kept

    def overlaps(self):
        """Пары частично перекрывающихся клипов на одной дорожке: (раньше, позже)."""
        by_track = {}
        for c in self.clips:
            by_track.setdefault(c.track, []).append(c)
        pairs = []
        for lst in by_track.values():
            lst.sort(key=lambda c: c.start)
            for i, a in enumerate(lst):
                for b in lst[i + 1:]:
                    if b.start >= a.end:
                        break
                    if b.end > a.end:          # вложенные клипы не кроссфейдим
                        pairs.append((a, b))
        return pairs

    def auto_fades(self):
        """{клип: (авто fade in, авто fade out)} для кроссфейдов в местах наложения."""
        res = {}
        if not self.auto_crossfade:
            return res
        for a, b in self.overlaps():
            ov = a.end - b.start
            ai, ao = res.get(a, (0, 0))
            res[a] = (ai, max(ao, ov))
            bi, bo = res.get(b, (0, 0))
            res[b] = (max(bi, ov), bo)
        return res

    def render(self, start, n):
        """Сводит все дорожки в стерео-буфер длиной n начиная с сэмпла start."""
        out = np.zeros((n, 2), dtype=np.float32)
        tracks = self.tracks
        clips = self.clips
        solo = any(t.solo for t in tracks)
        end = start + n
        auto = self.auto_fades()
        for c in clips:
            ti = c.track
            if ti < 0 or ti >= len(tracks):
                continue
            t = tracks[ti]
            if t.mute or (solo and not t.solo):
                continue
            cs, cl = c.start, c.length
            a, b = max(start, cs), min(end, cs + cl)
            if b <= a:
                continue
            off = c.src_start + (a - cs)
            seg = c.source.data[off: off + (b - a)]
            m = len(seg)
            if m == 0:
                continue
            g = db_to_lin(c.gain_db) * db_to_lin(t.volume_db)
            gains = np.array([g * min(1.0, 1.0 - t.pan), g * min(1.0, 1.0 + t.pan)], dtype=np.float32)
            seg = seg * gains
            fades = auto.get(c, (0, 0))
            if c.fade_in > 0 or c.fade_out > 0 or fades != (0, 0):
                p = np.arange(a - cs, a - cs + m, dtype=np.float32)
                seg *= clip_envelope(c, p, fades)[:, None]
            out[a - start: a - start + m] += seg
        np.clip(out, -1.0, 1.0, out=out)
        return out


# --------------------------------------------------------------------------- эффекты

def _atempo_chain(factor):
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.6f}")
    return parts


def fx_volume(data, gain_db=0.0, normalize=False, target_db=-1.0):
    if normalize:
        peak = float(np.abs(data).max()) if len(data) else 0.0
        if peak > 1e-9:
            gain_db = target_db - lin_to_db(peak)
    if abs(gain_db) < 1e-6:
        return data
    return filter_audio(data, f"volume={gain_db:.4f}dB")


def fx_pitch(data, semitones=0.0, tempo_pct=100.0, keep_duration=True):
    ratio = 2.0 ** (semitones / 12.0)
    filters = []
    if abs(semitones) > 1e-6:
        if keep_duration and ffmpeg_has("filters", "rubberband"):
            filters.append(f"rubberband=pitch={ratio:.6f}")
        else:
            filters += [f"asetrate={int(round(SR * ratio))}", f"aresample={SR}"]
            if keep_duration:
                filters += _atempo_chain(1.0 / ratio)
    if abs(tempo_pct - 100.0) > 1e-6:
        filters += _atempo_chain(tempo_pct / 100.0)
    return filter_audio(data, ",".join(filters))


EQ_BANDS = (31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def fx_eq(data, gains, highpass=None, lowpass=None):
    filters = []
    if highpass:
        filters.append(f"highpass=f={highpass:.1f}")
    if lowpass:
        filters.append(f"lowpass=f={lowpass:.1f}")
    for f, g in zip(EQ_BANDS, gains):
        if abs(g) > 0.01:
            filters.append(f"equalizer=f={f}:t=o:w=1:g={g:.2f}")
    return filter_audio(data, ",".join(filters))


def _fft_convolve(x, h):
    """Свёртка методом overlap-add (чтобы не выделять гигантские FFT на длинных клипах)."""
    lh = len(h)
    nfft = 1 << int(np.ceil(np.log2(max(2 * lh, 1 << 16))))
    block = nfft - lh + 1
    hf = np.fft.rfft(h, nfft)
    y = np.zeros(len(x) + lh - 1, dtype=np.float64)
    for i in range(0, len(x), block):
        seg = x[i:i + block]
        conv = np.fft.irfft(np.fft.rfft(seg, nfft) * hf, nfft)
        stop = min(i + nfft, len(y))
        y[i:stop] += conv[: stop - i]
    return y


def make_impulse_response(decay_s, predelay_ms, damping, width, seed=7):
    n = max(int(decay_s * SR), 256)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    env = 10.0 ** (-3.0 * t / decay_s)                      # -60 дБ к моменту decay_s
    hf_env = 10.0 ** (-3.0 * t / (decay_s * max(0.05, 1.0 - 0.9 * damping)))
    noise = rng.standard_normal((n, 2))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    lp = 1.0 / (1.0 + (freqs / 2500.0) ** 2)
    low = np.fft.irfft(np.fft.rfft(noise, axis=0) * lp[:, None], n, axis=0)
    ir = low * env[:, None] + (noise - low) * hf_env[:, None]
    attack = min(n, int(0.004 * SR))
    ir[:attack] *= np.linspace(0, 1, attack)[:, None]
    # ранние отражения
    for k in range(6):
        pos = int(rng.uniform(0.004, 0.035) * SR * (0.5 + decay_s / 4))
        if pos < n:
            ir[pos] += rng.uniform(0.5, 1.0) * (0.8 ** k) * np.sign(rng.standard_normal(2)) * 8
    mid = (ir[:, 0] + ir[:, 1]) / 2
    side = (ir[:, 0] - ir[:, 1]) / 2 * width
    ir = np.stack([mid + side, mid - side], axis=1)
    pre = int(predelay_ms * SR / 1000)
    if pre:
        ir = np.concatenate([np.zeros((pre, 2)), ir])
    ir /= np.sqrt((ir ** 2).sum(axis=0)).clip(min=1e-9)
    return ir


def fx_reverb(data, decay_s=2.0, predelay_ms=20.0, damping=0.5, wet=0.3, dry=1.0, width=1.0):
    """Свёрточная реверберация с синтетической импульсной характеристикой."""
    ir = make_impulse_response(decay_s, predelay_ms, damping, width)
    x = data.astype(np.float64)
    out = np.zeros((len(x) + len(ir) - 1, 2))
    for ch in range(2):
        out[:, ch] = _fft_convolve(x[:, ch], ir[:, ch]) * wet
    out[: len(x)] += x * dry
    loud = np.nonzero(np.abs(out).max(axis=1) > 1e-4)[0]
    stop = max(len(x), int(loud[-1]) + 1 if len(loud) else 0)
    return out[:stop].astype(np.float32)


NOISE_COLORS = {"Белый": "white", "Розовый": "pink", "Коричневый": "brown",
                "Синий": "blue", "Фиолетовый": "violet"}


def make_noise(color, n_samples, level_db):
    noise = generate_noise(color, n_samples).astype(np.float32)
    rms = float(np.sqrt(np.mean(noise ** 2))) or 1.0
    return noise * (db_to_lin(level_db) / rms)


def fx_noise(data, color="white", level_db=-30.0):
    return data + make_noise(color, len(data), level_db)


def fx_reverse(data):
    return np.ascontiguousarray(data[::-1])


def fx_channels(data, mode="mono"):
    left, right = data[:, 0], data[:, 1]
    if mode == "left":
        return np.stack([left, left], axis=1)
    if mode == "right":
        return np.stack([right, right], axis=1)
    if mode == "swap":
        return np.stack([right, left], axis=1)
    mono = (left + right) * 0.5
    return np.stack([mono, mono], axis=1)


# --------------------------------------------------------------------------- цепочка эффектов

FX = {
    "volume": fx_volume,
    "pitch": fx_pitch,
    "reverb": fx_reverb,
    "eq": fx_eq,
    "noise": fx_noise,
    "reverse": fx_reverse,
    "channels": fx_channels,
}

CHANNEL_MODES = {"left": "Только левый канал", "right": "Только правый канал",
                 "swap": "Поменять каналы местами", "mono": "Свести в моно"}


def make_effect(kind, **params):
    return {"kind": kind, "params": params, "enabled": True}


def render_chain(data, effects):
    for fx in effects:
        if fx.get("enabled", True):
            data = FX[fx["kind"]](data, **fx["params"])
    return data


def describe_effect(fx):
    k, p = fx["kind"], fx["params"]
    if k == "volume":
        if p.get("normalize"):
            return f"Нормализация до {p.get('target_db', -1):.1f} дБ"
        return f"Громкость {p.get('gain_db', 0):+.1f} дБ"
    if k == "pitch":
        parts = []
        if abs(p.get("semitones", 0)) > 1e-6:
            parts.append(f"тон {p['semitones']:+.2f} пт")
        if abs(p.get("tempo_pct", 100) - 100) > 1e-6:
            parts.append(f"темп {p['tempo_pct']:.0f}%")
        return "Тон/темп: " + (", ".join(parts) or "без изменений")
    if k == "reverb":
        return f"Реверберация {p.get('decay_s', 2):.1f} с, wet {p.get('wet', 0.3) * 100:.0f}%"
    if k == "eq":
        bands = sum(1 for g in p.get("gains", []) if abs(g) > 0.01)
        extra = (", HPF" if p.get("highpass") else "") + (", LPF" if p.get("lowpass") else "")
        return f"Эквалайзер (полос: {bands}{extra})"
    if k == "noise":
        name = next((n for n, c in NOISE_COLORS.items() if c == p.get("color")), str(p.get("color")))
        return f"Шум: {name.lower()}, {p.get('level_db', -30):.0f} дБ"
    if k == "reverse":
        return "Реверс"
    if k == "channels":
        return CHANNEL_MODES.get(p.get("mode"), "Каналы")
    return k


def short_effect_name(fx):
    return {"volume": "Гр", "pitch": "Тон", "reverb": "Рев", "eq": "EQ", "noise": "Шум",
            "reverse": "Реверс", "channels": "Кан"}.get(fx["kind"], fx["kind"])


# --------------------------------------------------------------------------- файл проекта

PROJECT_EXT = "mvproj"
PROJECT_VERSION = 1


def write_wav_f32(path, data):
    _run_ffmpeg(["-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", path],
                input_bytes=np.ascontiguousarray(data, dtype=np.float32).tobytes())


def save_project(project, path, view=None):
    """JSON со ссылками на исходные файлы и цепочками эффектов. Звук, которого нет
    в файлах (например, склеенные каналы), пишется в папку <проект>_media."""
    path = os.path.abspath(path)
    pdir = os.path.dirname(path)
    media = os.path.splitext(path)[0] + "_media"
    sources = {}
    for c in project.clips:
        base = c.fx_base if c.fx_base is not None else c.source
        if base.id in sources:
            continue
        if base.origin is None:
            os.makedirs(media, exist_ok=True)
            fpath = os.path.join(media, f"audio_{base.id}.wav")
            write_wav_f32(fpath, base.data)
            base.path = fpath
            base.origin = {"type": "file", "path": fpath}
        origin = dict(base.origin)
        if origin["type"] == "file":
            origin["path"] = os.path.abspath(origin["path"])
            try:
                origin["relpath"] = os.path.relpath(origin["path"], pdir)
            except ValueError:  # другой диск в Windows
                pass
        origin["name"] = base.name
        sources[base.id] = origin
    clips = []
    for c in project.clips:
        base = c.fx_base if c.fx_base is not None else c.source
        d = {"source": base.id, "track": c.track, "start": c.start, "src_start": c.src_start,
             "length": c.length, "gain_db": c.gain_db, "fade_in": c.fade_in,
             "fade_out": c.fade_out, "name": c.name}
        if c.fx_base is not None:
            d.update(fx_base_start=c.fx_base_start, fx_base_len=c.fx_base_len, effects=c.effects)
        clips.append(d)
    data = {"app": "Mini Vegas Audio", "version": PROJECT_VERSION, "sample_rate": SR,
            "auto_crossfade": project.auto_crossfade,
            "tracks": [vars(t) for t in project.tracks],
            "sources": {str(k): v for k, v in sources.items()}, "clips": clips, "view": view or {}}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_project(path, progress=None):
    """Возвращает (Project, view, список_ошибок)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version", 0) > PROJECT_VERSION:
        raise FFmpegError("Проект сохранён более новой версией программы")
    pdir = os.path.dirname(os.path.abspath(path))
    proj = Project()
    proj.auto_crossfade = data.get("auto_crossfade", True)
    for t in data.get("tracks", []):
        proj.tracks.append(Track(**t))
    proj._color_counter = max((t.color for t in proj.tracks), default=-1) + 1
    errors = []
    sources = {}
    items = list(data.get("sources", {}).items())
    for i, (sid, o) in enumerate(items):
        name = o.get("name", "?")
        try:
            if o["type"] == "silence":
                src = silence_source(o["n"], name)
            else:
                candidates = [os.path.join(pdir, o["relpath"])] if o.get("relpath") else []
                candidates.append(o["path"])
                fpath = next((p for p in candidates if os.path.isfile(p)), None)
                if fpath is None:
                    raise FFmpegError(f"файл не найден: {o['path']}")
                src = Source(decode_file(fpath), name, fpath)
            sources[sid] = src
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
        if progress:
            progress((i + 1) / max(len(items), 1) * 0.6)
    cache = {}
    clip_list = data.get("clips", [])
    for i, d in enumerate(clip_list):
        base = sources.get(str(d["source"]))
        if base is None:
            continue
        c = Clip(base, d["track"], d["start"], d["src_start"], d["length"], d.get("gain_db", 0.0),
                 d.get("fade_in", 0), d.get("fade_out", 0), d.get("name", base.name))
        effects = d.get("effects") or []
        if effects:
            bs, bl = d["fx_base_start"], d["fx_base_len"]
            key = effects_key(base, bs, bl, effects)
            if key not in cache:
                try:
                    cache[key] = Source(render_chain(base.data[bs:bs + bl], effects), base.name)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{c.name}: эффекты не применились ({e})")
                    cache[key] = None
            if cache[key] is None:
                continue
            c.source, c.fx_base, c.fx_base_start, c.fx_base_len = cache[key], base, bs, bl
            c.effects = effects
        c.src_start = max(0, min(c.src_start, c.source.length - 1))
        c.length = max(1, min(c.length, c.source.length - c.src_start))
        c.clamp_fades()
        if 0 <= c.track < len(proj.tracks):
            proj.clips.append(c)
        if progress:
            progress(0.6 + 0.4 * (i + 1) / max(len(clip_list), 1))
    return proj, data.get("view", {}), errors
