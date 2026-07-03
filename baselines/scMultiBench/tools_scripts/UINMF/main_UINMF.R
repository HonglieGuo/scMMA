library(rhdf5)
library(scran)
library(Seurat)
library(rliger)
library(HDF5Array)
library(magrittr)

# The script of UINMF for both vertical and cross-batch integration. 
# The output is joint embedding (dimension reduction)

write_h5 <- function(exprs_list, h5file_list) {
  if (length(unique(lapply(exprs_list, rownames))) != 1) {
    stop("rownames of exprs_list are not identical.")
  }
  
  for (i in seq_along(exprs_list)) {
    if (file.exists(h5file_list[i])) {
      warning("h5file exists! will rewrite it.")
      system(paste("rm", h5file_list[i]))
    }
    
    h5createFile(h5file_list[i])
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

begin_time <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)
n_batches <- (length(args) - 1) / 2
save_path <- args[length(args)]

rna_paths <- args[1:n_batches]
adt_paths <- args[(n_batches + 1):(2 * n_batches)]

# Load RNA matrices
rna_data <- list()
for (i in 1:n_batches) {
  temp <- h5_to_matrix(rna_paths[i])
  # Add batch prefix to avoid barcode collisions
  colnames(temp) <- paste0("batch", i, "_", colnames(temp))
  rna_data[[paste0("batch", i)]] <- temp
}

# Create Liger
lig <- createLiger(rna_data)
lig <- rliger::normalize(lig)
# selectGenes works on the entire lig object
lig <- selectGenes(lig, var.thresh = 0.1)
lig <- scaleNotCenter(lig)

# For unshared features
for (i in 1:n_batches) {
  adt <- h5_to_matrix(adt_paths[i])
  colnames(adt) <- paste0("batch", i, "_", colnames(adt))
  adt <- as(adt, "CsparseMatrix")
  
  # Seurat preprocessing to get variable unshared features and scaled data
  se <- CreateSeuratObject(counts = adt)
  se <- NormalizeData(se, verbose = FALSE)
  se <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  top2000 <- VariableFeatures(se)
  se <- ScaleData(se, features = top2000, verbose = FALSE)
  
  unshareScaled <- GetAssayData(se, layer = "scale.data")[top2000, ]
  
  # Assign to lig
  lig@var.unshared.features[[paste0("batch", i)]] <- top2000
  lig@scale.unshared.data[[paste0("batch", i)]] <- unshareScaled
}

# Run UINMF
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
