# tasting_analysis.py — backward-compatibility shim.
# All functions have been consolidated into analysis.py.
# Old notebooks that import from this module continue to work unchanged.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis import (
    processor_cross_correlation,
    run_ccf_workflow,
    summarize_ci,
    run_free_chemistry_workflow,
    run_equil_chemistry_workflow,
)

# Legacy alias kept for notebooks that used the old name
run_chemistry_workflow = run_free_chemistry_workflow
