#!/usr/bin/env python3
"""fft2 - FFT implementation."""
import sys,argparse,json,math
def fft(x):
    n=len(x)
    if n<=1:return x
    even=fft(x[0::2]);odd=fft(x[1::2])
    T=[complex(math.cos(-2*math.pi*k/n),math.sin(-2*math.pi*k/n))*odd[k] for k in range(n//2)]
    return [even[k]+T[k] for k in range(n//2)]+[even[k]-T[k] for k in range(n//2)]
def ifft(X):
    n=len(X);conj=[x.conjugate() for x in X]
    result=fft(conj)
    return [x.conjugate()/n for x in result]
def magnitude(c):return math.sqrt(c.real**2+c.imag**2)
def main():
    p=argparse.ArgumentParser(description="FFT")
    p.add_argument("values",nargs="+",type=float)
    p.add_argument("--inverse",action="store_true")
    args=p.parse_args()
    n=len(args.values);pad=1
    while pad<n:pad*=2
    data=[complex(v) for v in args.values]+[complex(0)]*(pad-n)
    if args.inverse:
        result=ifft(data)
        print(json.dumps({"inverse":[round(x.real,6) for x in result[:n]]}))
    else:
        result=fft(data)
        freqs=[{"bin":i,"magnitude":round(magnitude(r),6),"phase":round(math.atan2(r.imag,r.real),6)} for i,r in enumerate(result[:pad//2+1])]
        print(json.dumps({"n":n,"padded":pad,"frequencies":freqs},indent=2))
if __name__=="__main__":main()
