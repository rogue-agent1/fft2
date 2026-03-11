#!/usr/bin/env python3
"""FFT — Cooley-Tukey Fast Fourier Transform and applications.

Implements: radix-2 FFT/IFFT, power spectrum, convolution,
polynomial multiplication, frequency filtering, spectral analysis.

Usage: python fft2.py [--test]
"""

import sys, math, cmath

def fft(x):
    """Cooley-Tukey radix-2 DIT FFT. Input length must be power of 2."""
    N = len(x)
    if N <= 1:
        return list(x)
    if N & (N - 1):
        # Pad to next power of 2
        m = 1
        while m < N:
            m <<= 1
        x = list(x) + [0] * (m - N)
        N = m
    
    if N == 1:
        return [complex(x[0])]
    
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]

def ifft(X):
    """Inverse FFT."""
    N = len(X)
    # Conjugate, FFT, conjugate, scale
    conj = [x.conjugate() for x in X]
    result = fft(conj)
    return [x.conjugate() / N for x in result]

def power_spectrum(x):
    """Compute power spectral density |X(f)|²."""
    X = fft(x)
    return [abs(xi) ** 2 for xi in X]

def magnitude_spectrum(x):
    """Compute magnitude spectrum |X(f)|."""
    X = fft(x)
    return [abs(xi) for xi in X]

def phase_spectrum(x):
    """Compute phase spectrum arg(X(f))."""
    X = fft(x)
    return [cmath.phase(xi) for xi in X]

def convolve(a, b):
    """Fast convolution via FFT: a * b."""
    n = len(a) + len(b) - 1
    # Pad to power of 2
    m = 1
    while m < n:
        m <<= 1
    A = fft(list(a) + [0] * (m - len(a)))
    B = fft(list(b) + [0] * (m - len(b)))
    C = [ai * bi for ai, bi in zip(A, B)]
    result = ifft(C)
    return [x.real for x in result[:n]]

def polynomial_multiply(p, q):
    """Multiply polynomials using FFT. Coefficients in ascending order."""
    result = convolve(p, q)
    return [round(x) for x in result]

def low_pass_filter(signal, cutoff_bin):
    """Apply ideal low-pass filter in frequency domain."""
    X = fft(signal)
    N = len(X)
    for i in range(N):
        if cutoff_bin < i < N - cutoff_bin:
            X[i] = 0
    return [x.real for x in ifft(X)]

def find_dominant_frequency(signal, sample_rate):
    """Find the dominant frequency in a signal."""
    X = fft(signal)
    N = len(X)
    magnitudes = [abs(X[i]) for i in range(N // 2)]
    # Skip DC component
    max_idx = max(range(1, len(magnitudes)), key=lambda i: magnitudes[i])
    freq = max_idx * sample_rate / N
    return freq, magnitudes[max_idx]

def generate_sine(freq, duration, sample_rate, amplitude=1.0):
    """Generate sine wave samples."""
    N = int(sample_rate * duration)
    return [amplitude * math.sin(2 * math.pi * freq * i / sample_rate) for i in range(N)]

# --- Tests ---

def test_fft_single():
    result = fft([1])
    assert len(result) == 1
    assert abs(result[0] - 1) < 1e-10

def test_fft_dc():
    result = fft([1, 1, 1, 1])
    assert abs(result[0] - 4) < 1e-10
    for i in range(1, 4):
        assert abs(result[i]) < 1e-10

def test_fft_impulse():
    result = fft([1, 0, 0, 0])
    for x in result:
        assert abs(abs(x) - 1) < 1e-10

def test_ifft_roundtrip():
    signal = [1, 2, 3, 4, 5, 6, 7, 8]
    reconstructed = ifft(fft(signal))
    for a, b in zip(signal, reconstructed):
        assert abs(a - b.real) < 1e-10

def test_convolution():
    a = [1, 2, 3]
    b = [4, 5, 6]
    result = convolve(a, b)
    expected = [4, 13, 28, 27, 18]  # manual convolution
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6, f"{r} != {e}"

def test_polynomial_multiply():
    # (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
    p = [1, 2, 3]
    q = [4, 5]
    result = polynomial_multiply(p, q)
    assert result == [4, 13, 22, 15]

def test_parseval():
    """Parseval's theorem: sum|x|² = (1/N)sum|X|²."""
    signal = [1, -1, 2, -2, 3, -3, 4, -4]
    time_energy = sum(abs(x)**2 for x in signal)
    X = fft(signal)
    freq_energy = sum(abs(xi)**2 for xi in X) / len(X)
    assert abs(time_energy - freq_energy) < 1e-6

def test_dominant_frequency():
    sr = 1024
    signal = generate_sine(100, 1.0, sr)
    # Pad to power of 2
    freq, mag = find_dominant_frequency(signal, sr)
    assert abs(freq - 100) < 2, f"Expected ~100 Hz, got {freq}"

def test_linearity():
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]
    Fa = fft(a)
    Fb = fft(b)
    combined = fft([a[i] + b[i] for i in range(4)])
    for i in range(4):
        assert abs(combined[i] - (Fa[i] + Fb[i])) < 1e-10

if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_fft_single()
        test_fft_dc()
        test_fft_impulse()
        test_ifft_roundtrip()
        test_convolution()
        test_polynomial_multiply()
        test_parseval()
        test_dominant_frequency()
        test_linearity()
        print("All tests passed!")
