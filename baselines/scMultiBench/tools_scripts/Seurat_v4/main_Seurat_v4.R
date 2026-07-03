suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(rhdf5)
  library(HDF5Array)
})
options(future.globals.maxSize = 8000 * 1024^2)

# 动态获取当前脚本所在目录，防止因执行路径不同导致找不到 util.R
args_all <- commandArgs(trailingOnly = FALSE)
script_path <- grep("--file=", args_all, value = TRUE)
if (length(script_path) > 0) {
  script_dir <- dirname(sub("--file=", "", script_path[1]))
  source(file.path(script_dir, "util.R"))
} else {
  source("util.R")
}

# The script of Seurat WNN for vertical integration, RNA+ADT and RNA+ATAC data types. The output is joint embedding (dimension reduction)
# run commond for Seurat WNN (RNA+ADT)
# Rscript main_Seurat_v4.Rmd  '../../data/dataset_final/D3/rna.h5' '../../data/dataset_final/D3/adt.h5' NULL  '../../result/embedding/vertical integration/Seurat_WNN/D3/'
# run commond for Seurat WNN (RNA+ATAC)
#Rscript main_Seurat_v4.Rmd   '../../data/dataset_final/D15/rna.h5' NULL '../../data/dataset_final/D15/atac.h5' '../../result/embedding/vertical integration/Seurat_WNN/D15/'

# load parameters from
args <- commandArgs(trailingOnly = TRUE)
rna_paths_str <- args[1] 
adt_paths_str <- args[2] 
atac_paths_str <- args[3] 
save_path <- args[4] 

begin_time <- Sys.time()

rna_path <- if (rna_paths_str != "NULL") strsplit(rna_paths_str, ",")[[1]] else c("NULL")
adt_path <- if (adt_paths_str != "NULL") strsplit(adt_paths_str, ",")[[1]] else c("NULL")
atac_path <- if (atac_paths_str != "NULL") strsplit(atac_paths_str, ",")[[1]] else c("NULL")

if (rna_path[1] != "NULL" & adt_path[1] != "NULL" & atac_path[1] != "NULL"){
  result <- run_Seurat_RNA_ADT_ATAC(rna_path, adt_path, atac_path)
} else if (rna_path[1] != "NULL" & adt_path[1] != "NULL"){
  result <- run_Seurat_RNA_ADT(rna_path, adt_path)
} else if (rna_path[1] != "NULL" & atac_path[1] != "NULL"){
  result <- run_Seurat_RNA_ATAC(rna_path, atac_path)
} else if (adt_path[1] != "NULL" & atac_path[1] != "NULL"){
  result <- run_Seurat_ADT_ATAC(adt_path, atac_path)
}

dist <- result[['weighted.nn']]@nn.dist
idx <- result[['weighted.nn']]@nn.idx
end_time <- Sys.time()
all_time <- difftime(end_time, begin_time, units="secs")

if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
  print("path create")
}
write_h5(exprs_list = list(rna = dist), 
             h5file_list = c(paste0(save_path,"dist.h5")))
write_h5(exprs_list = list(rna = idx), 
             h5file_list = c(paste0(save_path,"idx.h5")))
             
# Save the WNN UMAP embedding to embedding.h5 for unified evaluation
emb <- result[['wnn.umap']]@cell.embeddings
write_h5(exprs_list = list(rna = emb),
             h5file_list = c(paste0(save_path, "embedding.h5")))

saveRDS(result[['weighted.nn']], file = paste0(save_path,"graph.rds"))
write.csv(all_time, paste0(save_path,"time.csv"))
