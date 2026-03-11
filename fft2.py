#!/usr/bin/env python3
"""FFT — Cooley-Tukey radix-2 Fast Fourier Transform."""
import cmath, math, sys

def fft(x):
    n = len(x)
    if n <= 1: return x
    if n & (n-1): raise ValueError("Length must be power of 2")
    even, odd = fft(x[::2]), fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + [even[k] - T[k] for k in range(n//2)]

def ifft(X):
    n = len(X)
    result = fft([x.conjugate() for x in X])
    return [x.conjugate() / n for x in result]

def magnitude(X): return [abs(x) for x in X]

def dominant_frequency(signal, sample_rate):
    X = fft(signal); n = len(X)
    mags = magnitude(X[:n//2])
    peak = max(range(1, len(mags)), key=lambda i: mags[i])
    return peak * sample_rate / n

if __name__ == "__main__":
    n = 256; sr = 1000; freq = 50
    signal = [math.sin(2*math.pi*freq*t/sr) + 0.5*math.sin(2*math.pi*120*t/sr) for t in range(n)]
    X = fft(signal)
    mags = magnitude(X[:n//2])
    peaks = sorted(range(1, len(mags)), key=lambda i: -mags[i])[:3]
    print("Top frequencies:")
    for p in peaks: print(f"  {p*sr/n:.0f} Hz (magnitude {mags[p]:.1f})")
    recon = ifft(X)
    err = max(abs(a - b.real) for a, b in zip(signal, recon))
    print(f"Reconstruction error: {err:.2e}")
