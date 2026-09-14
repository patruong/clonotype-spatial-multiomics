suppressPackageStartupMessages({
  library(Matrix)
})

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript export_infercnv_R_synthref_for_leiden.R <prepared_dir> <mode>")
}

prepared_dir <- args[[1]]
mode <- args[[2]]

outdir <- file.path(prepared_dir, paste0("infercnv_R_out_", mode))
export_dir <- file.path(outdir, "leiden_export")
dir.create(export_dir, recursive=TRUE, showWarnings=FALSE)

obj_candidates <- c(
  file.path(outdir, "run.final.infercnv_obj"),
  file.path(outdir, "run.final.infercnv_obj.rds"),
  file.path(outdir, "infercnv_obj.rds")
)

extra <- list.files(
  outdir,
  pattern="infercnv.*obj|run.final|\\.rds$",
  full.names=TRUE
)

obj_candidates <- unique(c(obj_candidates, extra))
obj_candidates <- obj_candidates[file.exists(obj_candidates)]

if (length(obj_candidates) == 0) {
  cat("[DEBUG] Files in outdir:\n")
  print(list.files(outdir, full.names=TRUE))
  stop("Could not find infercnv object in: ", outdir)
}

obj_path <- obj_candidates[[1]]
cat("[READ]", obj_path, "\n")

obj <- readRDS(obj_path)
expr <- obj@expr.data  # genes x cells/spots

if (!inherits(expr, "Matrix")) {
  expr <- Matrix(expr, sparse=TRUE)
}

# Remove synthetic reference columns; keep real spatial spots.
keep <- colnames(expr)[!grepl("^SYNTHREF", colnames(expr))]

anno_path <- file.path(prepared_dir, "annotations.tsv")
if (file.exists(anno_path)) {
  anno <- read.delim(
    anno_path,
    header=FALSE,
    stringsAsFactors=FALSE,
    col.names=c("barcode", "group")
  )
  query_barcodes <- anno$barcode[anno$group != "synthref"]
  keep2 <- query_barcodes[query_barcodes %in% colnames(expr)]
  if (length(keep2) > 0) {
    keep <- keep2
  }
}

expr <- expr[, keep, drop=FALSE]

writeMM(expr, file.path(export_dir, "infercnv_expr_genes_by_spots.mtx"))

write.table(
  rownames(expr),
  file.path(export_dir, "genes.tsv"),
  sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE
)

write.table(
  colnames(expr),
  file.path(export_dir, "spots.tsv"),
  sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE
)

write.table(
  colnames(expr),
  file.path(export_dir, "barcodes.tsv"),
  sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE
)

sp_path <- file.path(prepared_dir, "spatial_coords.tsv")
if (file.exists(sp_path)) {
  sp <- read.delim(sp_path, stringsAsFactors=FALSE)
  if ("barcode" %in% colnames(sp)) {
    rownames(sp) <- sp$barcode
    sp2 <- sp[colnames(expr), , drop=FALSE]
    sp2$barcode <- rownames(sp2)
    write.table(
      sp2,
      file.path(export_dir, "spatial_coords.tsv"),
      sep="\t", quote=FALSE, row.names=FALSE
    )
  }
}

cat("[OK] exported", nrow(expr), "genes x", ncol(expr), "spots/cells to", export_dir, "\n")
