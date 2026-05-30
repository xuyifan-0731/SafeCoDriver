# Search Protocol

This protocol adapts ARIS `/research-lit` and `/novelty-check` patterns to this topic.

## 1. Sources

Priority order:

1. Local project context: `/raid/xuyifan/jiqiuyu/docs/`, SafeCoDriver experiments/results.
2. ARIS helper search:
   - `arxiv_fetch.py`
   - `openalex_fetch.py`
   - `semantic_scholar_fetch.py` when not rate-limited.
3. Web search for exact paper titles, project pages, code, and newer surveys.

## 2. Query Families

### V2X Cooperative Perception Robustness

```text
V2X cooperative perception pose error robust spatial misalignment
robust cooperative perception autonomous driving noisy pose
V2X cooperative perception calibration error pose error
```

### Attacks and Faults

```text
adversarial attack cooperative perception autonomous driving V2X
sensor fault testing multi sensor fusion autonomous driving perception
Byzantine robust sensor fusion autonomous driving cooperative perception
trust management vehicular networks cooperative perception autonomous driving
```

### Downstream Safety

```text
cooperative perception safety autonomous driving planning collision
V2X cooperative autonomous driving accident prediction safety
perception error driving violations cooperative perception
```

## 3. Inclusion Criteria

Keep a paper if it directly addresses at least one:

- cooperative perception fusion under pose/time/calibration errors;
- V2X communication interruption, message loss, or bandwidth constraints;
- trust, reputation, malicious participant detection, Byzantine/fault-tolerant fusion;
- sensor fault injection and system-level autonomous-driving safety impact;
- datasets/benchmarks suitable for V2X anomaly injection.

## 4. Exclusion Criteria

Exclude or mark as background:

- generic autonomous-driving perception surveys with no V2X/fusion abnormality angle;
- communication-only VANET papers with no perception/fusion relevance;
- adversarial examples for standalone camera/VLM systems unless they inform attack taxonomy;
- pure mAP-improvement fusion backbones with no robustness or error handling.

## 5. Output Discipline

For each kept paper:

```text
title, year, venue/source, URL/arXiv ID, problem, method, overlap, gap for our work
```

Each novelty-risk item should map to a concrete prior paper.
