source('./Preparation/utils.R')
args <- commandArgs(trailingOnly = TRUE)

if ("--help" %in% args || length(args) == 0) {
  cat("
Usage: Rscript 2_combine_subsets.R [param1] [param2]

Description:
  This script used for qulity control

Arguments:
  filename     data filename (Required).
  task         experiment name (Required).

Example:
  Rscript Preparation/2_combine_subsets.R sln_sub10_demo.rds sln_sub10_transfer

")
  quit(status = 0)
}

if (length(args) >= 1) {
  filename <- args[1]
}
if (length(args) >= 2) {
  task <- args[2]
}


data_path <- './Example_data/'
data_file <- pj(data_path, filename)

# load data
data = readRDS(data_file)
obj_split <- SplitObject(data, split.by = "batch")

nfeature = 4000
samples <- tolower(unique(data@meta.data$batch))
modes <- c('rna', 'adt')


output_path <- pj("./result/preprocess", task)
print(output_path)
input_dirs <- pj(output_path, "inputdata", samples, "seurat")
output_feat_dir <- pj(output_path, "feat")
mkdir(output_feat_dir, remove_old = F)


adt_genes <- get_adt_genes()


merge_counts <- function(mode) {
  
    # Load different subsets
    prt("Processing ", mode, " data ...")
    sc_list <- list()
    feat_list <- list()
    subset_id <- "0"
    for (dataset_id in seq_along(input_dirs)) {
        print(dataset_id)
        file <- pj(input_dirs[dataset_id], paste0(mode, ".h5seurat"))
        if (file.exists(file)) {
           prt("Loading ", file, " ...")
           sc <- LoadH5Seurat(file, verbose = FALSE)
           cell_num <- dim(sc)[2]
           ids <- c(1, cell_num + 1)
           sc_list[[subset_id]] <- sc
           sc_list[[subset_id]]$subset_id <- subset_id
           if (mode == "rna") {
                sc_list[[subset_id]] <- remove_sparse_genes(sc_list[[subset_id]], kept_genes = adt_genes)
            }
           feat_list[[subset_id]] <- rownames(sc_list[[subset_id]])
           subset_id <- toString(strtoi(subset_id) + 1)
        }
    }
    # Combine features in different subsets
    feat_union <- Reduce(union, feat_list)

    # Remove low-frequency features
    mask_list <- list()
    mask_sum_list <- list()
    cell_num_total <- 0
    for (subset_id in names(feat_list)) {
        mask_list[[subset_id]] <- as.integer(feat_union %in% feat_list[[subset_id]])
        cell_num <- dim(sc_list[[subset_id]])[2]
        mask_sum_list[[subset_id]] <- mask_list[[subset_id]] * cell_num
        cell_num_total <- cell_num_total + cell_num
    }
    mask_sum_total <- Reduce(`+`, mask_sum_list)
    mask_ratio <- mask_sum_total / cell_num_total
    feat_union <- feat_union[mask_sum_total > 5000 | mask_ratio > 0.5]

    # Find highly variable features
    var_feat_list <- list()
    for (subset_id in names(sc_list)) {
        sc_list[[subset_id]] <- subset(sc_list[[subset_id]], features = feat_union)
        if (mode == "rna") {
            sc_list[[subset_id]] <- FindVariableFeatures(sc_list[[subset_id]], nfeatures = nfeature)
        } else if (mode == "adt") {
            VariableFeatures(sc_list[[subset_id]]) <- rownames(sc_list[[subset_id]])
        } else {
            stop(paste0(mode, ": Invalid modeality"))
        }
        var_feat_list[[subset_id]] <- VariableFeatures(sc_list[[subset_id]])
    }

    # Only keep features belong to the union of variable features
    var_feat_union <- Reduce(union, var_feat_list)
    if (mode == "rna") {
        var_feat_union <- union(var_feat_union, adt_genes)
    }
    sc_list <- lapply(sc_list, subset, features = var_feat_union)
    prt("Length of var_feat_union: ", length(var_feat_union))
    prt("Feature numbers of sc_list: ")
    lst <- lapply(sc_list, FUN = function(x) {length(rownames(x))})
    df <- as.data.frame(lst)
    colnames(df) <- names(lst)
    print(df)

    # Align features by merging different subsets, with missing features filled by zeros
    subset_num <- length(sc_list)
    if (subset_num > 1) {
        sc_merge <- merge(sc_list[[1]], unlist(sc_list[2:subset_num]),
            add.cell.ids = paste0("B", names(sc_list)), merge.data = T)
    } else {
        sc_merge <- RenameCells(sc_list[[1]], add.cell.id = paste0("B", names(sc_list)[1]))
    }
    feat_merged <- rownames(sc_merge)
    rownames(sc_merge[[mode]]@counts) <- feat_merged  # correct feature names for count data
 
    # Split into subsets and select features
    sc_split <- SplitObject(sc_merge, split.by = "subset_id")
    if (mode == "rna") {
        # Re-select 4000 variable features for each subset, rank all selected features, and keep
        # the top 4000 as the final variable features
        var_feat_integ <- SelectIntegrationFeatures(sc_split, fvf.nfeatures = nfeature, nfeatures = nfeature)
        var_feat_integ <- intersect(union(var_feat_integ, adt_genes), feat_merged)
    } else {
        var_feat_integ <- feat_merged
    }
    feat_dims[[mode]] <<- length(var_feat_integ)
    write.csv(var_feat_integ, file = pj(output_feat_dir, paste0("feat_names_", mode, ".csv")))
    sc_split <- lapply(sc_split, subset, features = var_feat_integ)

    # Get feature masks for each subset
    mask_list <- list()
    for (subset_id in names(sc_split)) {
        mask_list[[subset_id]] <- as.integer(var_feat_integ %in% rownames(sc_list[[subset_id]]))
    }
    prt("Feature numbers of sc_split: ")
    lst <- lapply(mask_list, sum)
    df <- as.data.frame(lst)
    colnames(df) <- names(lst)
    print(df)

    # Save subsets
    for (subset_id in names(sc_split)) {
        prt("Saving subset ", subset_id, " ...")
        sc_split[[subset_id]]@meta.data$cell_types <- obj_split[[as.integer(subset_id) + 1]]@meta.data$celltype

        # Save count data
        data <- t(data.matrix(sc_split[[subset_id]][[mode]]@counts))  # cell * var_gene
        output_exp_dir <- pj(output_path, paste0("subset_", subset_id), "mat")
        mkdir(output_exp_dir, remove_old = F)
        print(pj(output_exp_dir, paste0(mode, ".csv")))
        write.csv(data, file = pj(output_exp_dir, paste0(mode, ".csv")))

        # Save cell IDs
        write.csv(rownames(data), file = pj(output_path, paste0("subset_", subset_id), "cell_names.csv"))
        
        # Save labels
        if (mode == "rna") {
            write.csv(sc_split[[subset_id]]@meta.data, file = pj(output_path, paste0("subset_", subset_id), "labels.csv"))
        }

        # Save the feature mask
        output_mask_dir <- pj(output_path, paste0("subset_", subset_id), "mask")
        mkdir(output_mask_dir, remove_old = F)
        mask <- t(data.matrix(mask_list[[subset_id]]))  # 1 * D
        write.csv(mask, file = pj(output_mask_dir, paste0(mode, ".csv")))

    }
}


feat_dims <- list()
for (m in modes) {
    merge_counts(m)
}
# Save feature dimensionalities
prt("feat_dims: ", feat_dims)
write.csv(feat_dims, file = pj(output_feat_dir, "feat_dims.csv"))
