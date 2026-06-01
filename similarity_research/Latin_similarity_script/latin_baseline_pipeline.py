"""
===============================================================================
Add Latin Row - ScriptGap similarity_index_summary.csv
===============================================================================
Scores Latin with the ORIGINAL complexity formula (min-max normalized,
C = 0.5*E_norm + 0.3*V_norm + 0.2*F_norm, S = 1 - C) and appends it as one
row in the same column schema:

    script, similarity_S, complexity_C, median_expansion_ratio,
    median_vertical_footprint, median_friction_score, font_count

This does NOT re-score or re-center anything. The 8 existing rows are copied
through byte-for-byte; only the Latin row is computed and added. Because Latin
is interior to the existing min/max on all three axes, including it in the
normalization does not change any of the existing values.
===============================================================================
"""

import csv
import sys
from pathlib import Path

# --- Config -----------------------------------------------------------------
SUMMARY_IN  = "complexity_index_summary.csv"               # your file
SUMMARY_OUT = "complexity_index_summary_with_latin.csv"

# Latin baseline medians (from latin_baseline_summary.csv / latin_baseline.json).
# Swap in the full-precision E_latin_median etc. if you want exact digits.
E_LATIN, V_LATIN, F_LATIN = 1.4449, 1.25, 2.0
LATIN_FONT_COUNT = 2606          # nunique font_file, matches the other rows' method

W_E, W_V, W_F = 0.50, 0.30, 0.20
INCLUDE_LATIN_IN_NORMALIZATION = True   # Latin is interior, so this changes nothing
# ----------------------------------------------------------------------------


def main():
    p = Path(SUMMARY_IN)
    if not p.exists():
        sys.exit(f"Point SUMMARY_IN at your complexity_index_summary.csv (not found at {p})")

    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    Es = [float(r["median_expansion_ratio"]) for r in rows]
    Vs = [float(r["median_vertical_footprint"]) for r in rows]
    Fs = [float(r["median_friction_score"]) for r in rows]
    if INCLUDE_LATIN_IN_NORMALIZATION:
        Es += [E_LATIN]; Vs += [V_LATIN]; Fs += [F_LATIN]

    def nrm(x, xs):
        lo, hi = min(xs), max(xs)
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    En, Vn, Fn = nrm(E_LATIN, Es), nrm(V_LATIN, Vs), nrm(F_LATIN, Fs)
    C = W_E * En + W_V * Vn + W_F * Fn
    S = 1.0 - C

    latin_row = {
        "script": "Latin",
        "similarity_S": S,
        "complexity_C": C,
        "median_expansion_ratio": E_LATIN,
        "median_vertical_footprint": V_LATIN,
        "median_friction_score": F_LATIN,
        "font_count": LATIN_FONT_COUNT,
    }

    print("Latin normalized coords:")
    print(f"  E_norm={En:.6f}  V_norm={Vn:.6f}  F_norm={Fn:.6f}")
    print(f"  complexity_C={C}   similarity_S={S}\n")
    print("Latin row:")
    print(",".join(fields))
    print(",".join(str(latin_row[k]) for k in fields))

    with open(SUMMARY_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:          # existing scripts, verbatim
            w.writerow(r)
        w.writerow(latin_row)   # Latin appended
    print(f"\nWrote {len(rows) + 1} rows -> {SUMMARY_OUT}")


if __name__ == "__main__":
    main()