| setting | mode | WPC | Warn | wp_coll/wp_total | fake removal | real support | fake support | real trust support | fake trust support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count-min1 | fake_front | 1.800% | 100.00% | 72/4000 | 0.00% | 2.00 | 1.00 | 2.00 | 1.00 |
| count-min1 | noise+fake_front | 2.500% | 90.75% | 100/4000 | 0.00% | 1.47 | 1.00 | 1.47 | 1.00 |
| count-min2 | fake_front | 0.425% | 56.50% | 17/4000 | 100.00% | 2.00 | 1.00 | 2.00 | 1.00 |
| count-min2 | noise+fake_front | 0.450% | 56.75% | 18/4000 | 99.37% | 1.47 | 1.00 | 1.47 | 1.00 |
| trust-weighted | fake_front | 0.425% | 56.50% | 17/4000 | 100.00% | 2.00 | 1.00 | 1.20 | 0.20 |
| trust-weighted | noise+fake_front | 0.425% | 56.75% | 17/4000 | 99.37% | 1.47 | 1.00 | 0.88 | 0.20 |

Interpretation:

- Count-only `min_peer_support=1` is vulnerable to one colluding fake support source.
- Count-only `min_peer_support=2` rejects the fake object when only one support peer colludes.
- Trust-weighted evidence also rejects the fake object under `min_peer_support=1` when the colluding sender trust is low (`0.2`).
- This gives a clear paper boundary: the method is not robust to arbitrary collusion, but is robust when evidence thresholds or trust weights penalize colluding support.
