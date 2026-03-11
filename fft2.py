#!/usr/bin/env python3
"""FFT — Cooley-Tukey Fast Fourier Transform."""
import sys, cmath, math

def fft(x):
    n = len(x)
    if n <= 1: return x
    even = fft(x[0::2]); odd = fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + [even[k] - T[k] for k in range(n//2)]

def ifft(X):
    n = len(X)
    result = fft([x.conjugate() for x in X])
    return [x.conjugate() / n for x in result]

def magnitude(X):
    return [abs(x) for x in X]

def frequency_bins(n, sample_rate):
    return [k * sample_rate / n for k in range(n//2)]

if __name__ == "__main__":
    # Generate test signal: 50Hz + 120Hz
    n = 256; sr = 1000
    signal = [math.sin(2*math.pi*50*t/sr) + 0.5*math.sin(2*math.pi*120*t/sr) for t in range(n)]
    X = fft(signal)
    mags = magnitude(X)[:n//2]
    freqs = frequency_bins(n, sr)
    # Find peaks
    peaks = sorted(enumerate(mags), key=lambda x: -x[1])[:5]
    print(f"FFT of {n} samples at {sr}Hz")
    print(f"\nDominant frequencies:")
    for idx, mag in peaks:
        if mag > 1: print(f"  {freqs[idx]:6.1f} Hz  magnitude={mag:.2f}")
    # Verify roundtrip
    recovered = ifft(X)
    err = max(abs(signal[i] - recovered[i].real) for i in range(n))
    print(f"\nRoundtrip error: {err:.2e}")
