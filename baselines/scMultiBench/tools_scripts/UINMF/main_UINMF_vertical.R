
# for vertical integration, need to use the old versions
# devtools::install_github("welch-lab/liger", ref = "v1.0.1")
library(rhdf5)
library(scran)
library(Seurat)
library(rliger)
library(HDF5Array)
library(magrittr)

# The script of UINMF for vertical integration. The output is joint embedding (dimension reduction)
# run commond for UINMF
# Rscript main_UINMF_vertical.Rmd  '../../data/dataset_final/D3/rna.h5'  '../../data/dataset_final/D3/adt.h5' '../../result/embedding/diagonal integration/D3/UINMF/' 

write_h5 <- function(exprs_list, h5file_list) {
  if (length(unique(lapply(exprs_list, rownames))) != 1) {
    stop("rownames of exprs_list are not identical.")
  }
  
  for (i in seq_along(exprs_list)[1:(length(seq_along(exprs_list)))]) {
    if (file.exists(h5file_list[i])) {
      warning("h5file exists! will rewrite it.")
      system(paste("rm", h5file_list[i]))
    }
    
    h5createFile(h5file_list[i])
    #h5createGroup(h5file_list[i], "data")
    writeHDF5Array(((exprs_list[[i]])), h5file_list[i], name = "data")
    print(h5ls(h5file_list[i]))
  }
}


h5_to_matrix <- function(path){
    h5_data <- h5read(path,"matrix")
    feature <- h5_data$features
    barcode <- h5_data$barcodes
    data <- t(h5_data$data)
    colnames(data) <- as.character(barcode)
    rownames(data) <- as.character(feature)
    rownames(data) <- gsub("_", "-", rownames(data))
    return (data)
}

run_UINMF <- function(rna_path, adt_path, save_path){
  begin_time <- Sys.time()

  rna <- h5_to_matrix(rna_path)
  lig <- createLiger(list(rna = rna))
  lig <- normalize(lig) %>%
    selectGenes(var.thresh = 0.1) %>%
    scaleNotCenter()
  
  adt <- h5_to_matrix(adt_path)
  adt <- as(adt, "CsparseMatrix")
  se <- CreateSeuratObject(counts = adt)
  se <- NormalizeData(se, verbose = FALSE)
  se <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  top2000 <- VariableFeatures(se)
  se <- ScaleData(se, features = top2000, verbose = FALSE)
  unshareScaled <- GetAssayData(se, layer = "scale.data")[top2000, ]
  lig@var.unshared.features[["rna"]] <- top2000
  lig@scale.unshared.data[["rna"]] <- unshareScaled
  
  lig <- optimizeALS(lig, k = 30, max.iters = 30, use.unshared = TRUE)
  lig <- quantile_norm(lig)
  
  result <- lig@H.norm
  end_time <- Sys.time()
  all_time <- difftime(end_time, begin_time, units="secs")
  
  if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
    print("path create")
  }
  write_h5(exprs_list = list(rna = result), h5file_list = c(paste0(save_path,"embedding.h5")))
  write.csv(all_time, paste(save_path,"time.csv"))
}

# read parameters, the input is RNA counts and ATAC gene activity score counts
begin_time <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)
rna_path <- args[1] 
adt_path <- args[2] 
save_path <- args[3] 
run_UINMF(rna_path, adt_path, save_path)
