#!/usr/bin/env python3
"""Export stress-aligned digit samples as base64-encoded WAV for embedding in HTML."""

import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.effects import speedup
from gtts import gTTS
import io
import base64
import json

SAMPLE_RATE = 44100

def _find_stress_position(samples, sample_rate):
    window = int(0.020 * sample_rate)
    hop = window // 4
    rms = np.array([np.sqrt(np.mean(samples[i:i + window] ** 2))
                     for i in range(0, len(samples) - window, hop)])
    peak_idx = np.argmax(rms)
    return peak_idx * hop

def _strip_silence(audio, silence_thresh=-40, chunk_size=10):
    start = 0
    for i in range(0, len(audio), chunk_size):
        if audio[i:i + chunk_size].dBFS > silence_thresh:
            start = max(0, i - chunk_size)
            break
    end = len(audio)
    for i in range(len(audio) - chunk_size, 0, -chunk_size):
        if audio[i:i + chunk_size].dBFS > silence_thresh:
            end = min(len(audio), i + 2 * chunk_size)
            break
    return audio[start:end]

def main():
    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
        '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
        '8': 'eight', '9': 'nine'
    }

    raw_data = {}
    for d, word in digit_words.items():
        tts = gTTS(text=word, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio = AudioSegment.from_mp3(buf)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio = _strip_silence(audio)
        samp = np.array(audio.get_array_of_samples(), dtype=np.float64) / 32768.0
        stress_sample = _find_stress_position(samp, SAMPLE_RATE)
        stress_ms = stress_sample / SAMPLE_RATE * 1000
        raw_data[d] = (audio, stress_ms)
        print(f"  TTS '{word}': {len(audio)}ms, stress at {stress_ms:.0f}ms")

    target_ms = 400
    sped_up = {}
    for d, (seg, stress_ms) in raw_data.items():
        if len(seg) > target_ms + 30:
            ratio = len(seg) / target_ms
            seg_fast = speedup(seg, playback_speed=ratio, chunk_size=50, crossfade=25)
            samp = np.array(seg_fast.get_array_of_samples(), dtype=np.float64) / 32768.0
            new_stress = _find_stress_position(samp, SAMPLE_RATE) / SAMPLE_RATE * 1000
        else:
            seg_fast = seg
            new_stress = stress_ms
        sped_up[d] = (seg_fast, new_stress)

    STRESS_TARGET_MS = 80
    stress_overrides = {'9': -60, '1': -40, '6': -40, '8': -10}

    result = {}
    for d, (seg, stress_ms) in sped_up.items():
        word = digit_words[d]
        override = stress_overrides.get(d, 0)
        shift_ms = int(STRESS_TARGET_MS + override - stress_ms)

        if shift_ms > 0:
            seg = AudioSegment.silent(duration=shift_ms, frame_rate=SAMPLE_RATE) + seg
        elif shift_ms < 0:
            seg = seg[-shift_ms:]

        seg = seg[:target_ms]
        if len(seg) < target_ms:
            seg = seg + AudioSegment.silent(duration=target_ms - len(seg),
                                            frame_rate=SAMPLE_RATE)
        seg = seg.fade_out(40)

        # Export as individual WAV file
        seg.export(f'samples/{d}.wav', format='wav')

        # Export as WAV to base64
        wav_buf = io.BytesIO()
        seg.export(wav_buf, format="wav")
        wav_bytes = wav_buf.getvalue()
        b64 = base64.b64encode(wav_bytes).decode('ascii')
        result[d] = b64
        print(f"  '{word}': {len(wav_bytes)} bytes WAV, {len(b64)} chars base64")

    # Write as JS
    with open('samples/samples.js', 'w') as f:
        f.write('// Auto-generated digit samples (base64 WAV)\n')
        f.write('const DIGIT_SAMPLES = {\n')
        for d in sorted(result.keys()):
            f.write(f'  "{d}": "data:audio/wav;base64,{result[d]}",\n')
        f.write('};\n')

    total_size = sum(len(v) for v in result.values())
    print(f"\nTotal base64 size: {total_size / 1024:.0f} KB")
    print("Written to samples.js")

if __name__ == "__main__":
    main()
