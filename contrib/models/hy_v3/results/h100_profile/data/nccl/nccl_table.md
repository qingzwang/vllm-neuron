| 大小 | AR | AR(graph) | AG | AG(graph) | RS | RS(graph) |
|---|---|---|---|---|---|---|
| 16 KB | 49.24 | 15.67 | 48.82 | 10.13 | 48.52 | 9.80 |
| 32 KB | 49.69 | 17.74 | 49.28 | 12.34 | 48.79 | 10.60 |
| 64 KB | 50.08 | 17.94 | 50.80 | 12.45 | 49.96 | 10.72 |
| 128 KB | 51.07 | 18.64 | 52.77 | 12.62 | 51.68 | 10.84 |
| 256 KB | 53.56 | 19.99 | 55.35 | 12.68 | 53.10 | 11.10 |
| 512 KB | 56.01 | 20.08 | 58.39 | 13.12 | 54.66 | 12.44 |
| 1 MB | 59.24 | 28.40 | 60.30 | 17.01 | 58.25 | 15.95 |
| 2 MB | 51.19 | 41.76 | 61.14 | 22.32 | 58.21 | 20.34 |
| 16 MB | 126.04 | 122.45 | 71.95 | 69.37 | 75.53 | 72.32 |
| 64 MB | 325.81 | 322.47 | 209.70 | 198.60 | 202.69 | 199.27 |
| 256 MB | 1122.35 | 1119.39 | 727.52 | 701.16 | 723.62 | 719.97 |

all_reduce normal: 256MB out-of-place time 1122.35 us, algbw 239.17 GB/s, busbw 418.55 GB/s
all_reduce cudagraph: 256MB out-of-place time 1119.39 us, algbw 239.80 GB/s, busbw 419.66 GB/s
all_gather normal: 256MB out-of-place time 727.52 us, algbw 368.97 GB/s, busbw 322.85 GB/s
all_gather cudagraph: 256MB out-of-place time 701.16 us, algbw 382.85 GB/s, busbw 334.99 GB/s
reduce_scatter normal: 256MB out-of-place time 723.62 us, algbw 370.96 GB/s, busbw 324.59 GB/s
reduce_scatter cudagraph: 256MB out-of-place time 719.97 us, algbw 372.84 GB/s, busbw 326.24 GB/s
