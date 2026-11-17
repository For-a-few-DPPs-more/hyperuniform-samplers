# Stealthiness of the point sets

Every point set of the dataset was checked by computing, in float64, the structure factor

  S(**k**) = |Σⱼ exp(2iπ⟨**k**, **x**ⱼ⟩)|² / N

on **every** wave vector of its exclusion set: 304 point sets (`l2` and `wd` balls, D = 2–32, N = 2¹⁰–2¹⁷), 39,912,184 wave vectors in total. Both bounds below hold for all of them except 86 wave vectors (0.0002 %), described in [Exceptions](#exceptions).

## Uniform bound

Largest S(**k**) over all N, per dimension, exceptions (†) excluded.

| D | `l2` | `wd` |
|---:|---:|---:|
| 2 | 9.4·10⁻⁸ | 8.9·10⁻⁶ † |
| 3 | 2.9·10⁻⁹ | 1.6·10⁻⁵ † |
| 4 | 1.5·10⁻⁹ | 1.4·10⁻⁶ † |
| 5 | 8.6·10⁻¹⁰ | 2.1·10⁻⁶ † |
| 6 | 8.5·10⁻¹⁰ | 1.6·10⁻⁶ † |
| 7 | 8.7·10⁻¹⁰ | 2.4·10⁻⁶ |
| 8 | 8.7·10⁻¹⁰ | 2.2·10⁻⁶ |
| 9 | 9.0·10⁻¹⁰ | 1.3·10⁻⁶ |
| 10 | 9.3·10⁻¹⁰ | 8.6·10⁻⁷ |
| 11 | 8.8·10⁻¹⁰ | 8.0·10⁻⁷ |
| 12 | 9.1·10⁻¹⁰ | 6.5·10⁻⁷ |
| 13 | 9.1·10⁻¹⁰ | 7.2·10⁻⁷ |
| 14 | 9.5·10⁻¹⁰ | 3.6·10⁻⁷ |
| 15 | 9.3·10⁻¹⁰ | 1.9·10⁻⁷ |
| 16 | 8.7·10⁻¹⁰ | 3.1·10⁻⁷ |
| 20 | 8.8·10⁻¹⁰ | 2.7·10⁻⁷ |
| 24 | 9.0·10⁻¹⁰ | 1.1·10⁻⁷ |
| 28 | 9.0·10⁻¹⁰ | 6.7·10⁻⁸ |
| 32 | 9.1·10⁻¹⁰ | 4.0·10⁻⁸ |

## Scaling bound

  S(**k**) ≤ α N + β |**k**|²,  with α = 2·10⁻¹⁴ and β = 3·10⁻¹²

These are pessimistic worth case bounds, typical S(k) values are far below it.

## Exceptions

All 86 exceptions belong to the `wd` ball and are **axial** wave vectors (a single non-zero coordinate) of large norm, |**k**| ≥ 2608, for D ≤ 6 and N ≥ 2¹⁵ only. They exceed the scaling bound by up to a factor 18, with S(**k**) ≤ 8.8·10⁻⁴. They concentrate on three norms, around 2¹², 2¹³ and 2¹⁴/2π ≈ 2608, and when they occur in a point set they hit every coordinate axis. The phenomenon is not fully understood; it is most likely a numerical artifact of the float32 generation pipeline.

| \|**k**\| | count | D | N | max S | S / bound |
|---|---:|---|---|---:|---|
| 2608 | 29 | 2, 3, 4, 5, 6 | 2¹⁶, 2¹⁷ | 7.4·10⁻⁵ | 1.2 – 3.6 |
| 4095, 4097 | 42 | 2, 3, 4, 5 | 2¹⁵, 2¹⁶, 2¹⁷ | 8.8·10⁻⁴ | 3.6 – 17.5 |
| 8190–8194 | 15 | 2, 3 | 2¹⁷ | 3.2·10⁻⁴ | 1.0 – 1.6 |

Number of exceptions per point set:

| N | D | \|**k**\| = 2608 | \|**k**\| = 4095, 4097 | \|**k**\| = 8190–8194 | total |
|---:|---:|---:|---:|---:|---:|
| 2¹⁵ | 2 | – | 4 | – | 4 |
| 2¹⁶ | 2 | 2 | 4 | – | 6 |
| 2¹⁶ | 3 | 3 | 6 | – | 9 |
| 2¹⁶ | 4 | 4 | – | – | 4 |
| 2¹⁷ | 2 | 2 | 4 | 5 | 11 |
| 2¹⁷ | 3 | 3 | 6 | 10 | 19 |
| 2¹⁷ | 4 | 4 | 8 | – | 12 |
| 2¹⁷ | 5 | 5 | 10 | – | 15 |
| 2¹⁷ | 6 | 6 | – | – | 6 |

## Why this bound

The scaling bound has the form of the error of a float32 evaluation of S. The optimiser stalls at this level in float32 while it keeps improving in float64. : Numerically, rounding the phases ⟨**k**, **x**ⱼ⟩ contributes a term in ε_1*|**k**|², and accumulating the sum over N points contributes a term in ε_2*N.
