#!/usr/bin/env python3
"""fft2 - Fast Fourier Transform with inverse and convolution."""
import sys, cmath, math

def fft(x):
    n = len(x)
    if n <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

def ifft(X):
    n = len(X)
    conj = [x.conjugate() for x in X]
    result = fft(conj)
    return [x.conjugate() / n for x in result]

def convolve(a, b):
    n = 1
    while n < len(a) + len(b) - 1:
        n <<= 1
    fa = fft(a + [0] * (n - len(a)))
    fb = fft(b + [0] * (n - len(b)))
    fc = [fa[i] * fb[i] for i in range(n)]
    result = ifft(fc)
    return [round(x.real) for x in result[:len(a) + len(b) - 1]]

def test():
    # FFT of [1, 1, 1, 1] should be [4, 0, 0, 0]
    X = fft([1, 1, 1, 1])
    assert abs(X[0] - 4) < 1e-9
    for i in range(1, 4):
        assert abs(X[i]) < 1e-9
    # inverse
    x = ifft(X)
    for i in range(4):
        assert abs(x[i].real - 1) < 1e-9
    # convolution: [1,2,3] * [4,5] = [4, 13, 22, 15]
    assert convolve([1, 2, 3], [4, 5]) == [4, 13, 22, 15]
    # Parseval's theorem
    signal = [1, 2, 3, 4]
    X2 = fft(signal)
    energy_time = sum(abs(s)**2 for s in signal)
    energy_freq = sum(abs(f)**2 for f in X2) / len(signal)
    assert abs(energy_time - energy_freq) < 1e-9
    print("OK: fft2")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: fft2.py test")
