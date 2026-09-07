"""
Compare the Night 1 (2022-12-31, genuine, re-reduced 2026-07-02) and Night 2
(2023-01-01) chi Tau TELLURIC_DATA.fits molecfit outputs.

Purpose: judge whether the previous (erroneous) practice of reusing Night 2's
telluric template for Night 1 -- airmass-rescaled via Beer-Lambert only --
was a good-enough approximation, or whether it introduces a scientifically
significant error that requires re-running Night 1's telluric correction
(and everything downstream: A+B combination, flux calibration, retrieval).

Key comparison: the ACTUAL applied correction template, not just the raw
chi Tau spectra. Two candidates for "Night 1's telluric transmission at DH
Tau B's science airmass" are compared directly on a common wavelength grid:

  OLD (wrong):  T2(lambda) ** (X_DHTauB_N1 / X_chiTau_N2)   -- reuses Night 2's
                molecfit fit (different real atmosphere), only airmass-rescaled
  NEW (correct): T1(lambda) ** (X_DHTauB_N1 / X_chiTau_N1)  -- Night 1's own
                genuine molecfit fit, correctly airmass-rescaled

Beer-Lambert scaling only corrects for airmass (optical path length) under the
assumption that the absorber column density profile is IDENTICAL between the
two cases -- it cannot correct for a genuinely different atmospheric
composition (e.g. different precipitable water vapor). If PWV differs
significantly between nights, the OLD template will be systematically wrong
in H2O-dominated wavelength regions in a way that airmass rescaling cannot fix.
"""
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Airmass values (see observation_data_reduction_notes.md sections 2 and 8.2)
# ---------------------------------------------------------------------------
X_DHTauB_N1 = 1.7007     # DH Tau B science, Night 1
X_chiTau_N1 = 1.9050     # chi Tau, Night 1 (genuine, corrected 2026-07-02)
X_chiTau_N2 = 1.7251     # chi Tau, Night 2

alpha_new = X_DHTauB_N1 / X_chiTau_N1   # correct Night 1 airmass-rescaling factor
alpha_old = X_DHTauB_N1 / X_chiTau_N2   # what was actually used before the fix

print(f"alpha_new (correct, Night 1's own chi Tau) = {alpha_new:.4f}")
print(f"alpha_old (wrong, reused Night 2's chi Tau) = {alpha_old:.4f}")

# ---------------------------------------------------------------------------
# Load molecfit outputs
# ---------------------------------------------------------------------------
f1 = "/data2/peng/2022-12-31/chi_tau/2022-12-31/out/molecfit/TELLURIC_DATA.fits"
f2 = "/data2/peng/2023-01-01/chi_tau/2023-01-01/out/molecfit/TELLURIC_DATA.fits"
p1 = "/data2/peng/2022-12-31/chi_tau/2022-12-31/out/molecfit/BEST_FIT_PARAMETERS.fits"
p2 = "/data2/peng/2023-01-01/chi_tau/2023-01-01/out/molecfit/BEST_FIT_PARAMETERS.fits"

t1 = fits.getdata(f1)
t2 = fits.getdata(f2)
par1 = {str(row[0]): row[1] for row in fits.getdata(p1, 1)}
par2 = {str(row[0]): row[1] for row in fits.getdata(p2, 1)}

# ---------------------------------------------------------------------------
# 1. Fitted atmospheric composition: airmass alone cannot explain this
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MOLECFIT FITTED ATMOSPHERIC COMPOSITION (airmass-independent quantities)")
print("=" * 70)
keys = ["h2o_col_mm", "rel_mol_col_H2O", "rel_mol_col_CO2", "rel_mol_col_CH4",
        "rel_mol_col_N2O", "ppmv_H2O", "gaussfwhm", "reduced_chi2"]
print(f"{'parameter':<20}{'Night 1':>15}{'Night 2':>15}{'ratio N2/N1':>15}")
for k in keys:
    v1, v2 = par1[k], par2[k]
    ratio = v2 / v1 if v1 != 0 else np.nan
    print(f"{k:<20}{v1:>15.4f}{v2:>15.4f}{ratio:>15.3f}")

print(f"\n--> Precipitable water vapor: Night 1 = {par1['h2o_col_mm']:.3f} mm, "
      f"Night 2 = {par2['h2o_col_mm']:.3f} mm "
      f"({par2['h2o_col_mm']/par1['h2o_col_mm']:.2f}x more humid on Night 2).")
print("    This is a real change in atmospheric water content, not an airmass")
print("    effect -- Beer-Lambert airmass rescaling CANNOT correct for it.")

# ---------------------------------------------------------------------------
# 2. Interpolate both raw templates onto a common wavelength grid,
#    apply each night's OWN airmass-rescaling as it was/should be used,
#    and compare the two candidate "Night 1 science correction" templates.
# ---------------------------------------------------------------------------
wave_common = np.linspace(2063.7, 2472.4, 100_000)  # nm, science range only

# molecfit's own wavelength array is in microns; convert to nm
w1 = t1["lambda"] * 1e3
w2 = t2["lambda"] * 1e3
order1 = np.argsort(w1)
order2 = np.argsort(w2)
mtrans1 = np.interp(wave_common, w1[order1], t1["mtrans"][order1])
mtrans2 = np.interp(wave_common, w2[order2], t2["mtrans"][order2])

T_new = mtrans1 ** alpha_new   # correct: Night 1's own atmosphere, Night 1 airmass
T_old = mtrans2 ** alpha_old   # wrong:  Night 2's atmosphere, airmass-rescaled only

diff = T_new - T_old
abs_diff = np.abs(diff)

print("\n" + "=" * 70)
print("APPLIED CORRECTION TEMPLATE COMPARISON (science range 2063.7-2472.4 nm)")
print("=" * 70)
print(f"Mean |T_new - T_old|                 : {abs_diff.mean():.4f}")
print(f"Median |T_new - T_old|                : {np.median(abs_diff):.4f}")
print(f"95th percentile |T_new - T_old|        : {np.percentile(abs_diff, 95):.4f}")
print(f"Max |T_new - T_old|                    : {abs_diff.max():.4f}")
print(f"Fraction of pixels with |diff| > 0.05  : {(abs_diff > 0.05).mean()*100:.1f}%")
print(f"Fraction of pixels with |diff| > 0.10  : {(abs_diff > 0.10).mean()*100:.1f}%")
print(f"Fraction of pixels with |diff| > 0.20  : {(abs_diff > 0.20).mean()*100:.1f}%")

# Since the science pipeline DIVIDES the observed flux by the template
# (flux_corrected = flux_masked / mtrans_rescaled), what matters is the
# fractional error this would introduce into the corrected flux:
frac_err = (T_new - T_old) / T_new
print(f"\nFractional flux-correction error (T_new-T_old)/T_new if T_old had "
      f"been used for Night 1 as before:")
print(f"  Mean                                 : {frac_err.mean()*100:+.2f}%")
print(f"  Median                                : {np.median(frac_err)*100:+.2f}%")
print(f"  16th-84th percentile                  : "
      f"[{np.percentile(frac_err,16)*100:+.2f}%, {np.percentile(frac_err,84)*100:+.2f}%]")
print(f"  Fraction of pixels with |frac_err|>5%  : {(np.abs(frac_err) > 0.05).mean()*100:.1f}%")
print(f"  Fraction of pixels with |frac_err|>10% : {(np.abs(frac_err) > 0.10).mean()*100:.1f}%")

# Only unmasked (mtrans > 0.8, i.e. not already flagged out in section 8.4)
# pixels matter for the science spectrum -- deep telluric cores are masked
# to NaN regardless of which template is used.
unmasked = (T_new > 0.8) & (T_old > 0.8)
print(f"\nRestricted to pixels with mtrans > 0.8 in BOTH templates "
      f"(i.e. NOT masked by section 8.4; {unmasked.sum()}/{len(wave_common)} pixels):")
print(f"  Mean |T_new - T_old|                  : {abs_diff[unmasked].mean():.4f}")
print(f"  Mean fractional flux error             : {frac_err[unmasked].mean()*100:+.2f}%")
print(f"  Median fractional flux error           : {np.median(frac_err[unmasked])*100:+.2f}%")
print(f"  16th-84th percentile fractional error  : "
      f"[{np.percentile(frac_err[unmasked],16)*100:+.2f}%, "
      f"{np.percentile(frac_err[unmasked],84)*100:+.2f}%]")
print(f"  Fraction of unmasked pixels |err|>5%   : {(np.abs(frac_err[unmasked]) > 0.05).mean()*100:.1f}%")
print(f"  Fraction of unmasked pixels |err|>10%  : {(np.abs(frac_err[unmasked]) > 0.10).mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

ax = axes[0]
ax.plot(wave_common, T_new, lw=0.4, color="C0", label="T_new (Night 1 own chi Tau, correct)")
ax.plot(wave_common, T_old, lw=0.4, color="C1", alpha=0.7, label="T_old (Night 2 chi Tau, airmass-rescaled only)")
ax.set_ylabel("Transmission")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Applied telluric correction templates for Night 1 science airmass (X=1.7007)")

ax = axes[1]
ax.plot(wave_common, diff, lw=0.4, color="k")
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylabel("T_new - T_old")

ax = axes[2]
ax.plot(wave_common, frac_err * 100, lw=0.4, color="darkred")
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylabel("Fractional flux\ncorrection error (%)")
ax.set_ylim(-30, 30)

ax = axes[3]
ax.plot(wave_common, mtrans1, lw=0.4, color="C0", label="Night 1 raw mtrans (at chi Tau airmass 1.905)")
ax.plot(wave_common, mtrans2, lw=0.4, color="C1", alpha=0.7, label="Night 2 raw mtrans (at chi Tau airmass 1.725)")
ax.set_ylabel("Raw transmission")
ax.set_xlabel("Wavelength (nm)")
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
outpath = "/data2/peng/Recipe_DH_Tau_B/chi_tau_telluric_night1_vs_night2_comparison.png"
plt.savefig(outpath, dpi=150)
print(f"\nSaved comparison plot: {outpath}")
