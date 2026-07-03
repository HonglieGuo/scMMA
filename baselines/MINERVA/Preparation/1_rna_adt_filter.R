library(SeuratDisk)
source('./Preparation/utils.R')

args <- commandArgs(trailingOnly = TRUE)

if ("--help" %in% args || length(args) == 0) {
  cat("
Usage: Rscript 1_rna_adt_filter.R [param1] [param2]

Description:
  This script used for qulity control

Arguments:
  filename     data filename (Required).
  task         experiment name (Required).

Example:
  Rscript Preparation/1_rna_adt_filter.R dm_sub10_demo.rds dm_sub10

")
  quit(status = 0)
}

if (length(args) >= 1) {
  filename <- args[1]
}
if (length(args) >= 2) {
  task <- args[2]
}

base_path <- "./result/preprocess"
data_path <- "./Example_data/"
data_file <- pj(data_path, filename)
output_path <- pj(base_path, "/",task,"/inputdata")
mkdir(output_path, remove_old = T)


# load data
data = readRDS(data_file)
obj_split <- SplitObject(data, split.by = "batch")


for (batch in names(obj_split) ) {
    
    prt("Processing batch ", batch, " ...")
    output_dir <- pj(output_path, tolower(batch), "seurat")
    mkdir(output_dir, remove_old = T)
    
    rna_counts <- obj_split[[batch]]$RNA@counts
    adt_counts <- obj_split[[batch]]$ADT@counts
    # RNA
    rna <- gen_rna(rna_counts)
    # rna <- subset(rna, subset =
    #     nFeature_rna > 500 & nFeature_rna < 6000 & nCount_rna > 600 & nCount_rna < 40000 & percent.mt < 15)

    # ADT
    adt <- gen_adt(adt_counts)
    # adt <- subset(adt, subset = nCount_adt > 400 & nCount_adt < 20000)

    # Get intersected cells satisfying QC metrics of all modalities
    cell_ids <- Reduce(intersect, list(colnames(rna), colnames(adt)))
    rna <- subset(rna, cells = cell_ids)
    adt <- subset(adt, cells = cell_ids)
    
    # preprocess and save data
    preprocess(output_dir, rna = rna, adt = adt)
}
