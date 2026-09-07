#!/usr/bin/env python3
"""
tasting_combine_posteriors.py
==============================
Attempts to combine two independent MultiNest posterior distributions
(nights 2022-12-31 and 2023-01-01) via the product-of-posteriors formula:

    p(θ | D1, D2)  ∝  p(θ | D1) · p(θ | D2) / p(θ)

RESULT SUMMARY
--------------
The importance-sampling combination fails (ESS ≈ 1) because several
parameters are in significant inter-night tension:

  Parameter        Night1 med   Night2 med   Tension
  vsini            6.45 km/s    5.57 km/s    4.0 σ  ← physically fixed!
  nabla_RCE        0.125        0.064        4.5 σ  (T-P degeneracy)
  nabla_2          0.099        0.049        3.4 σ  (T-P degeneracy)
  log_P_RCE       -0.27        -1.28         3.2 σ  (T-P degeneracy)
  rv               31.61 km/s   31.92 km/s   2.9 σ  (wavelength cal?)
  T_bottom         3233 K       2657 K        2.6 σ  (T-P degeneracy)
  C/O              0.272        0.352        2.2 σ

The vsini tension (4σ) is physically impossible if DH Tau B is the same
object — it indicates the two nights' retrievals converge to different
T-P profile modes, and the vsini is degenerate with the T-P shape.
The rv tension suggests a possible inter-night wavelength calibration drift.

This output therefore provides:
  (A) A parameter tension table and ESS diagnostic
  (B) Per-parameter 1-D marginal products as an approximate estimate
      for parameters with reasonable overlap (FeH, log_g, log_12CO_13CO)
  (C) A comparison figure showing both nights individually (the most
      honest visualisation of the current state)

RECOMMENDED PATH FORWARD
-------------------------
Run a single joint retrieval on the two-night combined spectrum (Pathway 1):
  1. Apply per-chip wavelength-offset correction via CCF against telluric
     template (measures and removes inter-night instrumental drift)
  2. Co-add the two nights' spectra with LPU error propagation
  3. Run tasting_retrieval_equa_chem_v3.0.py on the combined data
This is the only rigorous approach when the single-night posteriors differ.

Outputs (all in OUT_DIR)
-------------------------
  combined_posteriors_diagnostics.txt         — tension table, ESS
  marginal_product_summary.json               — 1-D combined marginals
  comparison_figure_science_params.pdf        — 1-D comparison: night 1 vs 2
  comparison_cornerplot_individual_nights.pdf — corner overlay of both nights
"""

import json
import os

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

# ── Configuration ──────────────────────────────────────────────────────────
DIR1 = '/data2/peng/retrievals/2918065_N600_ev0.5_Normsavgol_PerChipScaleFalse/'
DIR2 = '/data2/peng/retrievals/3041995_N600_ev0.5_Normsavgol_PerChipScaleFalse/'
OUT_DIR = '/data2/peng/retrievals/combined_2918065_3041995/'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_MARGINAL_SAMPLES = 5000

# ── Load equal-weight posterior samples ────────────────────────────────────
# MultiNest format: columns [param_0, ..., param_16, log_likelihood]
s1 = np.loadtxt(DIR1 + 'pmn_post_equal_weights.dat')
s2 = np.loadtxt(DIR2 + 'pmn_post_equal_weights.dat')

theta1 = s1[:, :-1].copy()   # (N1, 17)
theta2 = s2[:, :-1].copy()   # (N2, 17)
N1, N2 = len(theta1), len(theta2)
n_params = theta1.shape[1]

assert n_params == 17, f'Expected 17 free parameters, got {n_params}'
print(f'Run 1 (2918065, night 2022-12-31): {N1} equal-weight samples')
print(f'Run 2 (3041995, night 2023-01-01): {N2} equal-weight samples')

# ── Parameter definitions ──────────────────────────────────────────────────
# Order matches free_params in tasting_retrieval_equa_chem_v3.0.py
param_keys = [
    'rv',           # 0  Gaussian prior N(32, 0.2²)
    'vsini',        # 1  uniform [0, 20]
    'log_g',        # 2  Gaussian prior N(3.5, 0.2²)
    'nabla_RCE',    # 3  uniform [0.04, 0.34]
    'nabla_0',      # 4  uniform [0.04, nabla_RCE]
    'nabla_1',      # 5  uniform [0.04, nabla_RCE]
    'nabla_2',      # 6  uniform [0.04, nabla_RCE]
    'nabla_3',      # 7  uniform [0.00, nabla_RCE]
    'nabla_4',      # 8  uniform [0.00, nabla_RCE]
    'nabla_5',      # 9  uniform [0.00, nabla_RCE]
    'T_bottom',     # 10 uniform [1500, 5000]
    'log_P_RCE',    # 11 uniform [-3, 1]
    'dlog_P_bot',   # 12 uniform [0.20, 1.60]
    'dlog_P_top',   # 13 uniform [0.20, 1.60]
    'FeH',          # 14 uniform [-2, 3]
    'C/O',          # 15 uniform [0.1, 1.0]
    'log_12CO_13CO',# 16 uniform [0, 6]
]

param_labels = [
    r'$v_{\rm rad}$ (km/s)',
    r'$v\sin i$ (km/s)',
    r'$\log g$',
    r'$\nabla_{\rm RCE}$',
    r'$\nabla_{0}$',
    r'$\nabla_{1}$',
    r'$\nabla_{2}$',
    r'$\nabla_{3}$',
    r'$\nabla_{4}$',
    r'$\nabla_{5}$',
    r'$T_{\rm bot}$ (K)',
    r'$\log P_{\rm RCE}$',
    r'$\Delta\log P_{\rm bot}$',
    r'$\Delta\log P_{\rm top}$',
    r'[Fe/H]',
    r'C/O',
    r'$\log\,^{12}$CO/$^{13}$CO',
]

# Gaussian priors (must be divided out when combining; uniform prior
# corrections are constants and cancel in the product formula)
gauss_priors = {
    0: (32.0, 0.2),   # rv
    2: (3.5,  0.2),   # log_g
}


# ── Step 1: Tension table ──────────────────────────────────────────────────
def posterior_stats(arr):
    p16, p50, p84 = np.percentile(arr, [16, 50, 84])
    return p50, (p50 - p16), (p84 - p50)   # median, lower_err, upper_err

print('\n── Inter-night parameter tension ─────────────────────────────────')
print(f'  {"Parameter":<20}  {"Night1 med":>12}  {"Night2 med":>12}  '
      f'{"tension":>9}')
print('  ' + '-' * 60)

tensions = {}
for k, key in enumerate(param_keys):
    m1, lo1, hi1 = posterior_stats(theta1[:, k])
    m2, lo2, hi2 = posterior_stats(theta2[:, k])
    sigma1 = (lo1 + hi1) / 2
    sigma2 = (lo2 + hi2) / 2
    sigma_comb = np.sqrt(sigma1**2 + sigma2**2)
    t = abs(m1 - m2) / sigma_comb if sigma_comb > 0 else 0.0
    tensions[key] = t
    flag = '  *** HIGH TENSION ***' if t > 3.0 else ('  * moderate' if t > 2.0 else '')
    print(f'  {key:<20}  {m1:>12.4f}  {m2:>12.4f}  {t:>8.2f}σ{flag}')

n_high_tension = sum(1 for t in tensions.values() if t > 3.0)
print(f'\n  {n_high_tension} parameters above 3σ inter-night tension')


# ── Step 2: ESS of full-17D importance sampling (diagnostic only) ──────────
def kish_ess(weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))

def log_weight_1d_kde(base, other):
    """
    Compute importance weights using per-parameter 1-D Gaussian KDE.
    In 17D this degrades to ESS≈1 when any dimension has poor overlap.
    """
    log_w = np.zeros(len(base))
    for k in range(base.shape[1]):
        kde_k = gaussian_kde(other[:, k])
        lw_k = np.log(kde_k(base[:, k]))
        lw_k = np.nan_to_num(lw_k, nan=-np.inf)
        log_w += lw_k
        if k in gauss_priors:
            mu, sigma = gauss_priors[k]
            log_w -= norm.logpdf(base[:, k], mu, sigma)
    finite = np.isfinite(log_w)
    if finite.any():
        log_w -= np.max(log_w[finite])
    w = np.where(np.isfinite(log_w), np.exp(log_w), 0.0)
    w_sum = w.sum()
    if w_sum > 0:
        w /= w_sum
    return w

print('\n── Importance-sampling ESS (diagnostic) ─────────────────────────')
w12 = log_weight_1d_kde(theta1, theta2)
w21 = log_weight_1d_kde(theta2, theta1)
ess_12 = kish_ess(w12)
ess_21 = kish_ess(w21)
print(f'  ESS (night1 base, night2 KDE): {ess_12:.1f} / {N1}  '
      f'({100*ess_12/N1:.2f}% efficiency)')
print(f'  ESS (night2 base, night1 KDE): {ess_21:.1f} / {N2}  '
      f'({100*ess_21/N2:.2f}% efficiency)')
if min(ess_12, ess_21) < 100:
    print('\n  *** ESS < 100: full posterior product has collapsed.')
    print('      The two-night posteriors do not overlap in 17-D space.')
    print('      Root cause: vsini (4σ) and nabla_RCE (4.5σ) tensions.')
    print('      A joint retrieval on the combined spectrum is required.')


# ── Step 3: Per-parameter 1-D marginal product (approximate) ──────────────
# p_comb_k(x) ∝ KDE1_k(x) · KDE2_k(x) / prior_k(x)
# Valid for individual parameter constraints; 2-D joint is NOT captured.
# Parameters with poor 1-D overlap will give very broad or zero product.

def marginal_1d_combine(samples1_k, samples2_k, gauss_prior=None,
                         n_samples=N_MARGINAL_SAMPLES, n_grid=2000):
    """
    Return samples from the 1-D marginal product posterior.
    Uses CDF-based sampling on a fine grid.
    """
    std1, std2 = samples1_k.std(), samples2_k.std()
    lo = min(samples1_k.min(), samples2_k.min()) - 2 * max(std1, std2)
    hi = max(samples1_k.max(), samples2_k.max()) + 2 * max(std1, std2)
    x = np.linspace(lo, hi, n_grid)

    p1 = gaussian_kde(samples1_k)(x)
    p2 = gaussian_kde(samples2_k)(x)

    if gauss_prior is not None:
        mu, sig = gauss_prior
        prior = np.maximum(norm.pdf(x, mu, sig), 1e-30)
        p_comb = p1 * p2 / prior
    else:
        p_comb = p1 * p2

    p_comb = np.maximum(p_comb, 0.0)
    total = np.trapz(p_comb, x)

    rng = np.random.default_rng(SEED)
    if total <= 0:
        # Posteriors do not overlap: return a sample from the flat prior
        return rng.uniform(lo, hi, n_samples)

    p_comb /= total
    dx = x[1] - x[0]
    cdf = np.cumsum(p_comb) * dx
    cdf = np.clip(cdf, 0.0, 1.0)

    u = rng.uniform(0.0, 1.0, n_samples)
    return np.interp(u, cdf, x)


print('\nComputing 1-D marginal products...')
marginal_samples = np.zeros((N_MARGINAL_SAMPLES, n_params))
for k in range(n_params):
    marginal_samples[:, k] = marginal_1d_combine(
        theta1[:, k], theta2[:, k],
        gauss_prior=gauss_priors.get(k, None),
    )

# Summary statistics for marginal products
print('\n── 1-D marginal product summary ──────────────────────────────────')
print(f'  {"Parameter":<20}  {"Night1":>22}  {"Night2":>22}  {"Marginal prod":>22}')
print('  ' + '-' * 93)
marginal_summary = {}
for k, key in enumerate(param_keys):
    m1, lo1, hi1 = posterior_stats(theta1[:, k])
    m2, lo2, hi2 = posterior_stats(theta2[:, k])
    mc, loc, hic = posterior_stats(marginal_samples[:, k])
    t = tensions[key]
    note = ' [high tension]' if t > 3.0 else ''
    print(f'  {key:<20}  {m1:>+8.4f}+{hi1:.4f}-{lo1:.4f}  '
          f'{m2:>+8.4f}+{hi2:.4f}-{lo2:.4f}  '
          f'{mc:>+8.4f}+{hic:.4f}-{loc:.4f}{note}')
    marginal_summary[key] = {
        'night1_med': m1, 'night1_lo': lo1, 'night1_hi': hi1,
        'night2_med': m2, 'night2_lo': lo2, 'night2_hi': hi2,
        'marginal_product_med': mc, 'marginal_product_lo': loc, 'marginal_product_hi': hic,
        'tension_sigma': tensions[key],
    }

# Save outputs
np.save(OUT_DIR + 'marginal_product_samples.npy', marginal_samples)
with open(OUT_DIR + 'marginal_product_summary.json', 'w') as f:
    json.dump(marginal_summary, f, indent=2)


# ── Step 4: Write diagnostics text file ───────────────────────────────────
with open(OUT_DIR + 'combined_posteriors_diagnostics.txt', 'w') as f:
    f.write('DH Tau B — Combined posterior diagnostics\n')
    f.write('Nights: 2022-12-31 (2918065) × 2023-01-01 (3041995)\n')
    f.write(f'N1 = {N1},  N2 = {N2}\n\n')
    f.write('Product-of-posteriors (17-D importance sampling)\n')
    f.write(f'  ESS (1→2): {ess_12:.1f} / {N1}  '
            f'ESS (2→1): {ess_21:.1f} / {N2}\n')
    f.write('  Conclusion: ESS≈1; full posterior product has collapsed.\n\n')
    f.write('Inter-night tensions:\n')
    f.write(f'  {"Parameter":<20}  {"Night1 med":>12}  {"Night2 med":>12}  {"tension":>10}\n')
    f.write('  ' + '-' * 60 + '\n')
    for k, key in enumerate(param_keys):
        m1 = np.median(theta1[:, k])
        m2 = np.median(theta2[:, k])
        t = tensions[key]
        flag = ' ***' if t > 3.0 else (' *' if t > 2.0 else '')
        f.write(f'  {key:<20}  {m1:>12.4f}  {m2:>12.4f}  {t:>9.2f}σ{flag}\n')
    f.write('\n')
    f.write('Recommendation: run joint retrieval on combined spectrum\n')
    f.write('(wavelength-offset-corrected two-night co-add → Pathway 1)\n')

print(f'\nSaved diagnostics → {OUT_DIR}combined_posteriors_diagnostics.txt')
print(f'Saved marginal samples → {OUT_DIR}marginal_product_samples.npy')
print(f'Saved JSON summary → {OUT_DIR}marginal_product_summary.json')


# ── Step 5: Figures ────────────────────────────────────────────────────────
# Key science parameters (exclude T-P gradient params for clarity)
sci_keys = ['rv', 'vsini', 'log_g', 'FeH', 'C/O', 'log_12CO_13CO']
sci_idx  = [param_keys.index(k) for k in sci_keys]
sci_labels = [param_labels[i] for i in sci_idx]

BINS = 45
ALPHA = 0.55

# Figure A: 1-D comparison panels for all science parameters
fig_comp, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.ravel()

for i, (idx, key, label) in enumerate(zip(sci_idx, sci_keys, sci_labels)):
    ax = axes[i]
    t = tensions[key]
    is_high = t > 3.0

    ax.hist(theta1[:, idx], bins=BINS, density=True, alpha=ALPHA,
            color='C0', label=r'Night 1 (2022-12-31)')
    ax.hist(theta2[:, idx], bins=BINS, density=True, alpha=ALPHA,
            color='C1', label=r'Night 2 (2023-01-01)')
    ax.hist(marginal_samples[:, idx], bins=BINS, density=True, alpha=ALPHA,
            color='k', histtype='step', lw=1.5,
            label=r'1-D marginal product')

    m1 = np.median(theta1[:, idx])
    m2 = np.median(theta2[:, idx])
    mc = np.median(marginal_samples[:, idx])
    ax.axvline(m1, color='C0', lw=0.8, ls='--')
    ax.axvline(m2, color='C1', lw=0.8, ls='--')
    ax.axvline(mc, color='k',  lw=0.8, ls=':')

    title_color = 'firebrick' if is_high else 'k'
    ax.set_title(f'{label}\ninter-night tension: {t:.1f}σ',
                 fontsize=9, color=title_color)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel('Probability density', fontsize=8)
    if i == 0:
        ax.legend(fontsize=7)

fig_comp.suptitle(
    'DH Tau B — Inter-night posterior comparison (2022-12-31 × 2023-01-01)\n'
    'Black step = 1-D marginal product (valid for low-tension parameters only)',
    fontsize=10, y=1.02)
fig_comp.tight_layout()
out_comp = OUT_DIR + 'comparison_figure_science_params.pdf'
fig_comp.savefig(out_comp, bbox_inches='tight')
print(f'Saved comparison figure → {out_comp}')


# Figure B: Corner plot with BOTH nights overlaid (no forced combination)
CORNER_OPTS = dict(
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt='.3f',
    title_kwargs={'fontsize': 7},
    label_kwargs={'fontsize': 8},
    hist_kwargs={'alpha': 0.0},          # suppress base histogram
    contourf_kwargs={'alpha': 0.25},
    contour_kwargs={'alpha': 0.7},
)

fig_corner = corner.corner(
    theta1[:, sci_idx],
    labels=sci_labels,
    color='C0',
    **CORNER_OPTS,
)
corner.corner(
    theta2[:, sci_idx],
    fig=fig_corner,
    color='C1',
    **CORNER_OPTS,
)

# Marginal product on diagonal only (1D is reliable)
# Draw as a thin black step on each diagonal panel
n_sci = len(sci_idx)
for i in range(n_sci):
    ax_diag = fig_corner.axes[i * n_sci + i]
    h, bins = np.histogram(marginal_samples[:, sci_idx[i]], bins=40, density=True)
    centres = 0.5 * (bins[:-1] + bins[1:])
    ax_diag.plot(centres, h, color='k', lw=1.2, ls='-', alpha=0.9,
                 label='1-D marginal product' if i == 0 else None)

h1 = mlines.Line2D([], [], color='C0', lw=1.5, label='Night 1 — 2022-12-31 (2918065)')
h2 = mlines.Line2D([], [], color='C1', lw=1.5, label='Night 2 — 2023-01-01 (3041995)')
hm = mlines.Line2D([], [], color='k',  lw=1.5, ls='-',
                   label='1-D marginal product (diagonal only)')
fig_corner.legend(handles=[h1, h2, hm], loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), fontsize=8)
fig_corner.suptitle(
    'DH Tau B — Science parameter posteriors, both nights\n'
    'Off-diagonal: individual nights only (joint combination not available; see diagnostics)',
    y=1.02, fontsize=9)
out_corner = OUT_DIR + 'comparison_cornerplot_individual_nights.pdf'
fig_corner.savefig(out_corner, bbox_inches='tight')
print(f'Saved corner plot → {out_corner}')

plt.close('all')
print('\nDone.')
