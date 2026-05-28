"""ELANG — Electronic Enforcement & Analysis for Next-Gen traffic Governance.

Reference architecture from competition-analysis/DISHUB_Case_Analysis.md.
Phase 1 MVP (detection + stats) ships in this package. Phase 2 modules
(ANPR, tracking, heatmap, officer optimizer) live under `elang.stubs/`
and are now wired against real libraries — install the relevant
optional deps to enable each. Phase 3 (CRM classifier) remains stubbed.
"""

__version__ = "0.2.0"
