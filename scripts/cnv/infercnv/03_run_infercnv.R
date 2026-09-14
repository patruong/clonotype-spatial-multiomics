#!/usr/bin/env Rscript

options(scipen = 100)

suppressPackageStartupMessages({
  library(Matrix)
  library(infercnv)
})

set.seed(0)

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 3) {
  stop("Usage: run_infercnv_R_nonecrosis_one_sample_mode.R <prepared_sample_dir> <threads> <mode:noHMM|HMM_i6>")
}

prepared_dir <- normalizePath(args[[1]], mustWork=TRUE)
threads <- as.integer(args[[2]])
mode <- args[[3]]

if (!(mode %in% c("noHMM", "HMM_i6"))) {
  stop("mode must be noHMM or HMM_i6")
}

out_dir <- file.path(prepared_dir, paste0("infercnv_R_out_", mode))
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

counts_path <- file.path(prepared_dir, "counts.mtx")
genes_path <- file.path(prepared_dir, "genes.tsv")
barcodes_path <- file.path(prepared_dir, "barcodes.tsv")
anno_path <- file.path(prepared_dir, "annotations.tsv")
gene_order_path <- file.path(prepared_dir, "gene_order.tsv")
ref_path <- file.path(prepared_dir, "ref_group_names.txt")

cat("[INFO] prepared_dir =", prepared_dir, "\n")
cat("[INFO] out_dir      =", out_dir, "\n")
cat("[INFO] mode         =", mode, "\n")

counts <- Matrix::readMM(counts_path)
genes <- read.table(genes_path, sep="\t", stringsAsFactors=FALSE)[,1]
barcodes <- read.table(barcodes_path, sep="\t", stringsAsFactors=FALSE)[,1]

counts <- as.matrix(counts)
rownames(counts) <- genes
colnames(counts) <- barcodes

cat("[INFO] counts matrix dim =", nrow(counts), "genes x", ncol(counts), "spots\n")

ref_groups <- character(0)
if (file.exists(ref_path)) {
  ref_groups <- readLines(ref_path, warn=FALSE)
  ref_groups <- ref_groups[nchar(ref_groups) > 0]
}

if (length(ref_groups) == 0) {
  cat("[WARN] No reference groups retained. Running inferCNV without explicit references.\n")
  ref_groups_arg <- NULL
} else {
  cat("[INFO] Reference groups:\n")
  print(ref_groups)
  ref_groups_arg <- ref_groups
}

infercnv_obj <- infercnv::CreateInfercnvObject(
  raw_counts_matrix = counts,
  annotations_file = anno_path,
  delim = "\t",
  gene_order_file = gene_order_path,
  ref_group_names = ref_groups_arg
)

use_hmm <- mode == "HMM_i6"

infercnv_obj <- infercnv::run(
  infercnv_obj,
  cutoff = 0.1,
  out_dir = out_dir,
  cluster_by_groups = TRUE,
  denoise = TRUE,
  HMM = use_hmm,
  HMM_type = if (use_hmm) "i6" else NULL,
  analysis_mode = "samples",
  num_threads = threads,
  no_plot = FALSE
)

saveRDS(infercnv_obj, file=file.path(out_dir, "run.final.infercnv_obj.rds"))

expr <- infercnv_obj@expr.data
cnv_abs_mean <- Matrix::colMeans(abs(expr), na.rm=TRUE)

scores <- data.frame(
  barcode = names(cnv_abs_mean),
  infercnv_abs_mean = as.numeric(cnv_abs_mean),
  mode = mode,
  stringsAsFactors=FALSE
)

write.table(
  scores,
  file=file.path(out_dir, "infercnv_abs_mean_by_spot.tsv"),
  sep="\t",
  quote=FALSE,
  row.names=FALSE
)

cat("[DONE] wrote inferCNV output to", out_dir, "\n")
