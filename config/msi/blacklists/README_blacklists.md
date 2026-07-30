# MSI matrix / technical peak blacklists

These files are **inputs** to the clean NMF workflow. The clean pipeline does not recreate them automatically.

## 9-AA blacklist

File:

blacklists/9AA_matrix_exclude_mz_nmf_informed_v2.txt

Used for:

- bc2059
- bc2075

Matrix / polarity:

- 9-AA / 9-aminoacridine
- negative mode

Provenance:

1. Started from conservative matrix/adduct filtering based on Met-ID review and known 9-AA matrix-related peaks.
2. Ran RMS-normalized MSI NMF.
3. Inspected recurrent top-loading m/z families that repeatedly dominated multiple factors and looked technical/adduct-like rather than biological.
4. Added those recurrent families to an NMF-informed v1 blacklist.
5. Re-ran NMF/K-sweep.
6. Added residual recurrent families to create the current v2 blacklist.
7. Froze the v2 file for reproducible exploratory analysis.

Important:

This is not an "annotated-only" filter. Peaks were not removed just because they lacked Met-ID annotations. Peaks were removed because they behaved like recurrent matrix/adduct/technical families across NMF factors.

Current bc2059 behavior:

- selected model: K=4
- selected RGB: NMF1 / NMF3 / NMF4
- omitted factor: residual technical/high-mass questionable factor
- blacklist removed 106 measured MSI features in the clean bc2059 run

## DHBA blacklist

File:

blacklists/DHBA_matrix_exclude_mz.txt

Used for:

- bc2004
- bc2020

Matrix / polarity:

- DHB / DHBA
- positive mode

Provenance:

This is currently the DHBA conservative blacklist. It has **not yet** undergone the same NMF-informed v2 curation as 9-AA.

Do not reuse the 9-AA v2 blacklist for DHBA. DHBA has different matrix chemistry, adduct behavior, and polarity.

Future DHBA workflow:

1. Run DHBA with the conservative DHBA blacklist.
2. Inspect selected factors for recurrent technical/adduct families.
3. If needed, create:

   blacklists/DHBA_matrix_exclude_mz_nmf_informed_v2.txt

4. Re-run only DHBA samples with that DHBA-specific v2 blacklist.
