# MSI NMF filtering notes: tissue mask, necrosis removal, and matrix/adduct blacklists

Date: 2026-05-27

Working folder:

/Users/patricktruong/git/sma-vdj/scripts/nmf_remake_20260521/nmf_msi_filtered_technical_qc

Main output root:

/Users/patricktruong/git/sma-vdj/results/nmf_rerun_20260521

This note documents the current exploratory MSI NMF preprocessing/QC workflow for four adjacent SMA-VDJ sections:

bc2059  9AA   V13Y10-038_B1
bc2004  DHBA  V13Y10-060_B1

bc2075  9AA   V13Y10-038_D1
bc2020  DHBA  V13Y10-060_D1

The goal is to make MSI NMF factors easier to interpret by removing obvious background spots, necrosis/dead-cell regions, and recurrent matrix/adduct-like peaks before factorization.

---

## 1. Two-step spot filtering

The current pipeline uses two spot filters before NMF.

### Step 1: keep spots over tissue

The first filter removes empty/background spots and keeps only measured MSI spots that overlap tissue.

This is handled inside:

msi_rms_nmf.py

Relevant output:

spot_mask_qc/

Meaning of run labels:

withnecrosis
  Spots over tissue, but no pathology/necrosis exclusion.

nonecrosis_knn
  Spots over tissue, then necrosis/dead-cell spots removed using ST-to-MSI kNN pathology-label transfer.

### Step 2: remove necrosis/dead-cell regions by ST-to-MSI kNN

Important: this does not impute MSI intensities onto ST spots.

Instead:

ST pathology labels + ST spatial coordinates
  -> kNN label transfer
  -> measured MSI spots receive estimated necrosis fraction
  -> MSI spots predicted to overlap necrosis/dead-cell regions are removed before NMF

Current arguments:

--pathology_annot_dir /Users/patricktruong/git/sma-vdj/data/all_annotations
--exclude_pathology_regex "necros|necro|necrot|dead|död|nekro"
--spaceranger_out /Users/patricktruong/git/sma-vdj/data/space_ranger_outs
--pathology_match_mode barcode_or_knn
--pathology_knn_k 3
--pathology_knn_weight_power 2.0
--pathology_knn_necrosis_threshold 0.5
--require_pathology_annotations

Pathology annotation files:

/Users/patricktruong/git/sma-vdj/data/all_annotations

Useful QC output:

pathology_qc/*pathology_filter_summary.txt
pathology_qc/*pathology_knn_transfer.tsv
pathology_qc/*st_pathology_labels_used_for_knn.tsv

Important bug fix:

correct capture ID: V13Y10-038_D1
wrong capture ID:   V13Y10-038-D1

The wrong ID caused kNN transfer to fail because it searched for a non-existent Space Ranger folder. After fixing this, bc2075 correctly used:

/Users/patricktruong/git/sma-vdj/data/space_ranger_outs/V13Y10-038_D1/spatial

Example successful bc2075 pathology filtering:

capture_id: V13Y10-038_D1
knn_removed_spots: 289
combined_removed_spots: 289
spots_before: 2444
spots_after: 2155

---

## 2. Matrix/adduct blacklist filtering

NMF is run after matrix-specific m/z blacklists are applied.

This means blacklisted peaks are removed from the MSI matrix before NMF.

The blacklists are pragmatic NMF-cleaning tools. They are not proof that every excluded peak is non-biological.

### 9AA blacklist

Current 9AA blacklist:

blacklists/9AA_matrix_exclude_mz_nmf_informed_v2.txt

Constructed from:

- known 9AA matrix/technical peaks
- manually observed NMF-dominant residual matrix/adduct families
- repeated peaks that appeared across factors and made NMF less interpretable

Purpose:

remove obvious 9AA matrix/adduct/background families while preserving metabolite/lipid biology.

Important caution:

Do not blacklist every unannotated peak. Many real MSI features are unannotated at 2 ppm.

### DHBA blacklist v2

Initial conservative DHBA blacklist:

blacklists/DHBA_matrix_exclude_mz_nmf_informed_v2.txt

This removed obvious DHBA matrix/adduct peaks, but several recurrent families still dominated NMF factors.

### DHBA blacklist v3 recurrent-family cleaner

Cleaner DHBA blacklist:

blacklists/DHBA_matrix_exclude_mz_nmf_informed_v3_recurrent_family.txt

This extends v2 by removing recurrent DHBA-dominant families observed after the conservative run.

Main suspected low-mass DHBA/matrix/isotope-like family:

169.076
170.079
170.084
171.092

Main suspected recurrent high-mass lipid/adduct/isotope family:

703.572
725.531 / 725.553
734.566
758.566 / 759.570 / 760.581
780.548 / 781.551 / 782.564
786.597
796.522 / 797.524 / 798.538

Rationale:

780.5478 - 758.5656 approx 21.982 Da

This is close to the Na-H adduct shift, suggesting some of these peaks may be related adduct/isotope family members rather than independent biological features.

Important caution:

v2 = conservative, preserves more lipid signal
v3 = cleaner, reduces recurrent adduct/lipid-family dominance

For current exploratory DHBA figures, v3 looked more interpretable.

---

## 3. K-sweep and factor selection

For each sample/matrix/mask condition, NMF is run across K values and a selector chooses a usable RGB combination.

Current K range:

9AA:  K=3..8
DHBA: K=3..8

The selector uses a blacklist-derived technical score.

technical_score means:

how much a factor's top loadings overlap with the current matrix-specific technical/adduct blacklist

Low technical score means the factor does not strongly overlap the current blacklist. It does not prove that the factor is biological.

Useful outputs:

k_sweep_factor_scores.tsv
k_sweep_k_decisions.tsv
k_sweep_factor_technical_scores.png
k_sweep_n_clean_factors.png
k_sweep_selected_technical_scores.png
best_k_selected_factors.txt
*_bestK*_rgb_nontechnical3.png
*_top_factor_loadings.png
factor_annotation_shortlists_2ppm/*compact.txt

---

## 4. Current preferred output folders

For 9AA, use v2 blacklist + nonecrosis kNN:

k_sweep_bc2059_9AA_nmf_informed_v2_nonecrosis_knn_threshold038/selected_model_outputs
k_sweep_bc2075_9AA_nmf_informed_v2_nonecrosis_knn_threshold038/selected_model_outputs

For DHBA, use v3 recurrent-family cleaner + nonecrosis kNN:

k_sweep_bc2004_DHBA_nmf_informed_v3_recurrent_family_nonecrosis_knn_threshold038/selected_model_outputs
k_sweep_bc2020_DHBA_nmf_informed_v3_recurrent_family_nonecrosis_knn_threshold038/selected_model_outputs

Open one at a time:

OUT="/Users/patricktruong/git/sma-vdj/results/nmf_rerun_20260521"

open "$OUT/k_sweep_bc2059_9AA_nmf_informed_v2_nonecrosis_knn_threshold038/selected_model_outputs"
open "$OUT/k_sweep_bc2004_DHBA_nmf_informed_v3_recurrent_family_nonecrosis_knn_threshold038/selected_model_outputs"
open "$OUT/k_sweep_bc2075_9AA_nmf_informed_v2_nonecrosis_knn_threshold038/selected_model_outputs"
open "$OUT/k_sweep_bc2020_DHBA_nmf_informed_v3_recurrent_family_nonecrosis_knn_threshold038/selected_model_outputs"

---

## 5. Current short biological interpretation

### Adjacent pair 1

bc2059 9AA  V13Y10-038_B1
bc2004 DHBA V13Y10-060_B1

Observation:

bc2059 NMF1 and bc2059 NMF4 spatially colocalize with bc2004 NMF3.

Interpretation:

bc2004 DHBA NMF3 appears to mark a broader tissue compartment.

In adjacent bc2059 9AA, this same region appears split into at least two biochemical subprograms:

- bc2059 NMF1: phosphometabolic / membrane-rich program
- bc2059 NMF4: ascorbate-like / antioxidant / redox-associated program

Preferred cautious wording:

A spatially conserved compartment is detected across adjacent 9AA and DHBA sections. The compartment appears to combine membrane/phosphometabolic chemistry with an ascorbate/redox-associated subprogram.

Do not claim the two matrices detect the exact same chemical program. Safer wording:

same anatomical/spatial compartment, matrix-specific chemical readouts

### Adjacent pair 2

bc2075 9AA  V13Y10-038_D1
bc2020 DHBA V13Y10-060_D1

Observation:

Both sections show broad compartment structure after necrosis removal. bc2075 had real kNN necrosis removal. bc2020 DHBA looked cleaner after v3 recurrent-family filtering.

Possible interpretation:

bc2075 and bc2020 likely capture related tissue architecture, but the factor correspondence is less direct than in the B1 pair.

bc2075 NMF3 remains worth manual inspection because it is dominated by a tight low-mass cluster:

160.842 / 162.839 / 164.836

This may be a real localized chemical niche or a residual structured technical family.

---

## 6. Met-ID caveat

The 2 ppm Met-ID tables should be used as family-level support, not exact compound identification.

Good wording:

ascorbate-like
ADP/ATP-like
acylcarnitine-like
phospholipid-like
lipid/adduct family

Avoid overclaiming exact molecular identity without MS/MS or stronger orthogonal evidence.

---

## 7. Relevance for future gene NMF, sVDJ NMF, and MOFA-FLEX

The same two-step spot filtering is probably useful before other unsupervised models.

Recommended general preprocessing:

1. restrict to spots over tissue
2. remove spots assigned to necrosis/dead-cell pathology
3. keep a QC table documenting which spots were removed and why

This should be considered before:

gene expression NMF
spatial VDJ / sVDJ NMF
MOFA-FLEX multimodal integration

Reason:

necrosis spots can create strong degradation/background factors that dominate unsupervised models

For MOFA-FLEX, a shared nonecrosis spot universe should ideally be used across modalities to avoid learning necrosis/dead-cell biology or degradation as a major latent factor.

Suggested future direction:

create a shared nonecrosis spot mask per section and apply it consistently to:
- ST gene expression
- MSI matrix
- spatial BCR/TCR/sVDJ matrices
- MOFA-FLEX inputs

For ST/gene/sVDJ data, pathology labels may be directly available by barcode and should be easier to apply than MSI. For MSI, the current approach transfers ST labels to MSI spots by spatial kNN.

---

## 8. Quick QC commands

Check pathology filter summaries:

OUT="/Users/patricktruong/git/sma-vdj/results/nmf_rerun_20260521"

find "$OUT" -type f \
  -path "*nonecrosis_knn*/pathology_qc/*pathology_filter_summary.txt" \
  -print -exec cat {} \;

Check selected factors:

OUT="/Users/patricktruong/git/sma-vdj/results/nmf_rerun_20260521"

for f in "$OUT"/k_sweep_*_nonecrosis_knn_threshold038/best_k_selected_factors.txt; do
  echo
  echo "==== $f ===="
  cat "$f"
done

---

## 9. How to rerun the current preferred pipeline

Run from:

/Users/patricktruong/git/sma-vdj/scripts/nmf_remake_20260521/nmf_msi_filtered_technical_qc

Commands:

bash run_mask_compare_v2_qc.sh 9aa
bash run_mask_compare_v2_qc.sh dhba

Or run both:

bash run_mask_compare_v2_qc.sh all

Current preferred setup:

9AA:
  samples: bc2059, bc2075
  blacklist: blacklists/9AA_matrix_exclude_mz_nmf_informed_v2.txt
  K sweep: 3..8
  outputs: withnecrosis and nonecrosis_knn

DHBA:
  samples: bc2004, bc2020
  blacklist: blacklists/DHBA_matrix_exclude_mz_nmf_informed_v3_recurrent_family.txt
  K sweep: 3..8
  outputs: withnecrosis and nonecrosis_knn

Before rerunning, check DHBA uses v3:

grep -n "DHBA_matrix" run_mask_compare_v2_qc.sh

Expected:

blacklists/DHBA_matrix_exclude_mz_nmf_informed_v3_recurrent_family.txt
