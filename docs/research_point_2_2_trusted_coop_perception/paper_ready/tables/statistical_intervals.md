| setting | method | wp_coll/wp_total | WPC | Wilson 95% CI |
| --- | --- | ---: | ---: | ---: |
| synthetic-full/clean | MultiPeerObjectGuard | 51/17530 | 0.291% | [0.221%, 0.382%] |
| synthetic-full/drop | MultiPeerObjectGuard | 51/17530 | 0.291% | [0.221%, 0.382%] |
| synthetic-full/fake_front | MultiPeerObjectGuard | 51/17530 | 0.291% | [0.221%, 0.382%] |
| synthetic-full/noise+fake_front | MultiPeerObjectGuard | 51/17530 | 0.291% | [0.221%, 0.382%] |
| synthetic-v1-20x20/clean | MultiPeerObjectGuard | 17/4000 | 0.425% | [0.266%, 0.680%] |
| synthetic-v1-20x20/noise+fake_front | MultiPeerObjectGuard | 17/4000 | 0.425% | [0.266%, 0.680%] |
| real-single-full | EgoOnly | 331/17530 | 1.888% | [1.697%, 2.100%] |
| real-single-full | CleanCoop oracle | 51/17530 | 0.291% | [0.221%, 0.382%] |
| real-single-full | RealOtherRaw | 277/17530 | 1.580% | [1.406%, 1.776%] |
| real-single-full | RealOtherTrustCalib | 277/17530 | 1.580% | [1.406%, 1.776%] |
| real-single-full | RealOtherObjectGuard | 331/17530 | 1.888% | [1.697%, 2.100%] |
| real-multi-20x20 | EgoOnly | 74/4000 | 1.850% | [1.476%, 2.316%] |
| real-multi-20x20 | CleanCoop oracle | 17/4000 | 0.425% | [0.266%, 0.680%] |
| real-multi-20x20 | RealPrimaryTrustCalib | 69/4000 | 1.725% | [1.365%, 2.177%] |
| real-multi-20x20 | RealMultiEvidenceGuard | 65/4000 | 1.625% | [1.277%, 2.066%] |

Notes:

- These are Wilson binomial confidence intervals over waypoint-collision counts. They are a compact paper-table sanity check, not a replacement for scenario-level paired bootstrap.
- For the strongest synthetic claim, the full-val final method gives identical WPC for clean, dropout, fake-front, and noise+fake-front modes.
- For real multi-source, the interval overlap between `RealPrimaryTrustCalib` and `RealMultiEvidenceGuard` shows that the current 20x20 real improvement should be presented as preliminary and mechanism-diagnostic, not as a strong final real-data superiority claim.
