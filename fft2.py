import argparse, cmath, math

def fft(x):
    n = len(x)
    if n <= 1: return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    t = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

def ifft(x):
    n = len(x)
    conj = [v.conjugate() for v in x]
    result = fft(conj)
    return [v.conjugate() / n for v in result]

def main():
    p = argparse.ArgumentParser(description="FFT tool")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--values", nargs="+", type=float)
    p.add_argument("--freq", type=float, help="Generate sine wave")
    p.add_argument("-n", "--samples", type=int, default=64)
    args = p.parse_args()
    if args.freq:
        signal = [math.sin(2 * math.pi * args.freq * t / args.samples) for t in range(args.samples)]
        result = fft(signal)
        print("Top frequencies:")
        mags = [(abs(result[k]), k) for k in range(args.samples // 2)]
        mags.sort(reverse=True)
        for mag, k in mags[:5]:
            if mag > 0.1: print(f"  bin {k}: magnitude={mag:.4f}")
    elif args.values:
        n = len(args.values)
        pad = 1
        while pad < n: pad *= 2
        vals = args.values + [0] * (pad - n)
        result = fft(vals)
        for i, v in enumerate(result[:len(args.values)]):
            print(f"  [{i}] {abs(v):.4f} ∠{cmath.phase(v)*180/cmath.pi:.1f}°")
    elif args.demo:
        signal = [math.sin(2*math.pi*3*t/64) + 0.5*math.sin(2*math.pi*7*t/64) for t in range(64)]
        result = fft(signal)
        print("Signal: sin(3x) + 0.5*sin(7x)")
        mags = [(abs(result[k]), k) for k in range(32)]
        mags.sort(reverse=True)
        for mag, k in mags[:5]:
            if mag > 0.5: print(f"  freq bin {k}: {mag:.2f}")
    else: p.print_help()

if __name__ == "__main__":
    main()
