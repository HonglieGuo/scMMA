library(argparse)
library(Seurat)
library(SeuratDisk)
library(future)
library(dplyr)
library(Matrix)
library(purrr)
library(stringr)
library(RcppTOML)
set.seed(1234)

pj <- file.path


prt <- function(...) {
    cat(paste0(..., "\n"))
}


mkdir <- function(directory, remove_old = F) {
    if (remove_old) {
        if (dir.exists(directory)) {
             prt("Removing directory ", directory)
             unlink(directory, recursive = T)
        }
    }
    if (!dir.exists(directory)) {
        dir.create(directory, recursive = T)
    }
}


mkdirs <- function(directories, remove_old = F) {
    for (directory in directories) {
        mkdir(directory, remove_old = remove_old)
    }
}


random_round <- function(mat) {
    mat_floor <- floor(mat)
    res <- mat - mat_floor
    res[] <- rbinom(n = nrow(res) * ncol(res), size = 1, prob = res)
    mat <- mat_floor + res
    mode(mat) <- "integer"
    return(mat)
}


# creat Seurat Object
gen_rna <- function(rna_counts, min_cells = 3) {
    rna <- CreateSeuratObject(
        counts = rna_counts,
        min.cells = min_cells,
        assay = "rna"
    )
    rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^MT-")
    return(rna)
}


remove_sparse_genes <- function(obj, assay = "rna", min_cell_percent = 1, kept_genes = NULL) {
    assay_ <- DefaultAssay(obj)
    DefaultAssay(obj) <- assay
    min_cells <- 0.01 * min_cell_percent * ncol(obj)
    mask <- rowSums(obj[[assay]]@counts > 0) > min_cells & rowSums(obj[[assay]]@counts) > 2 * min_cells
    feats <- rownames(obj[[assay]]@counts)[mask]
    if (!is.null(kept_genes)) {
        feats <- union(feats, kept_genes)
    }
    obj <- subset(obj, features = feats)
    DefaultAssay(obj) <- assay_
    return(obj)
}


gen_adt <- function(adt_counts) {
    # rename features
    feat <- unlist(map(rownames(adt_counts), tolower))
    feat <- unlist(map(feat, gsub, pattern = "-|_|\\(|\\)|/", replacement = "."))
    feat <- unlist(map(feat, gsub, pattern = "^cd3$", replacement = "cd3.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd4$", replacement = "cd4.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd11b$", replacement = "cd11b.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd26$", replacement = "cd26.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd38$", replacement = "cd38.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.recombinant$", replacement = "cd56.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd57.recombinant$", replacement = "cd57"))
    feat <- unlist(map(feat, gsub, pattern = "^cd90.thy1.$", replacement = "cd90"))
    feat <- unlist(map(feat, gsub, pattern = "^cd112.nectin.2.$", replacement = "cd112"))
    feat <- unlist(map(feat, gsub, pattern = "^cd117.c.kit.$", replacement = "cd117"))
    feat <- unlist(map(feat, gsub, pattern = "^cd138.1.syndecan.1.$", replacement = "cd138.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd155.pvr.$", replacement = "cd155"))
    feat <- unlist(map(feat, gsub, pattern = "^cd269.bcma.$", replacement = "cd269"))
    feat <- unlist(map(feat, gsub, pattern = "^clec2$", replacement = "clec1b"))
    feat <- unlist(map(feat, gsub, pattern = "^cadherin11$", replacement = "cadherin"))
    feat <- unlist(map(feat, gsub, pattern = "^folate.receptor$", replacement = "folate"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.1$", replacement = "notch1"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.2$", replacement = "notch3"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.a.b$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.2$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.g.d$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.1$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va7.2$", replacement = "tcr.v.7.2"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va24.ja18$", replacement = "tcr.v.24.j.18"))
    feat <- unlist(map(feat, gsub, pattern = "^vegfr.3$", replacement = "vegfr3"))
    rownames(adt_counts) <- feat
    # remove features
    adt_counts <- adt_counts[!grepl("^igg", rownames(adt_counts)), ]
    # create adt object
    adt <- CreateSeuratObject(
      counts = adt_counts,
      assay = "adt"
    )
    return(adt)
}


preprocess <- function(output_dir, atac = NULL, rna = NULL, adt = NULL) {
    # preprocess and save data

    if (!is.null(rna)) {
        rna <- NormalizeData(rna) %>%
               FindVariableFeatures(nfeatures = 4000) %>%
               ScaleData()
        SaveH5Seurat(rna, pj(output_dir, "rna.h5seurat"), overwrite = TRUE)
    }

    if (!is.null(adt)) {
        VariableFeatures(adt) <- rownames(adt)
        adt <- NormalizeData(adt, normalization.method = "CLR", margin = 2) %>%
               ScaleData()
        SaveH5Seurat(adt, pj(output_dir, "adt.h5seurat"), overwrite = TRUE)
    }
}


get_adt_genes <- function(file_path = "./Preparation/adt_rna_correspondence.csv") {
    adt_genes_raw <- read.csv(file_path, sep = "\t")[["symbol"]]
    adt_genes <- vector()
    for (gene in adt_genes_raw) {
        if (gene %in% c("not_found", "")) {
            next
        } else if (grepl(",", gene)) {
            adt_genes <- c(adt_genes, strsplit(gene, split = ",")[[1]])
        } else {
            adt_genes <- c(adt_genes, gene)
        }
    }
    return(unique(adt_genes))
}
