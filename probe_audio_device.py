#!/usr/bin/env python3
"""Probe which sample rates a microphone actually accepts for input.

Usage:
    python3 probe_audio_device.py            # probe config.DEVICE_INDEX
    python3 probe_audio_device.py <device>   # probe a specific device index
"""

import sys

import sounddevice as sd

import config

CANDIDATE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else config.DEVICE_INDEX
    info = sd.query_devices(device)
    print(f"Device {device}: {info['name']}")
    print(f"  max input channels : {info['max_input_channels']}")
    print(f"  default samplerate  : {info['default_samplerate']:.0f} Hz (PortAudio suggestion, not exhaustive)")
    print()
    print("Testing candidate rates (channels=1):")
    for rate in CANDIDATE_RATES:
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            print(f"  {rate:>6} Hz : OK")
        except Exception as exc:
            print(f"  {rate:>6} Hz : FAILS -> {exc}")


if __name__ == "__main__":
    main()

