suppressPackageStartupMessages({
  library(Signac)
  library(Seurat)
  library(dplyr)
  library(rhdf5)
  library(HDF5Array)
})

h5_to_matrix <- function(path){
    h5_data <- h5read(path,"matrix") 
    feature <- h5_data$features
    barcode <- h5_data$barcodes
    data <- t(h5_data$data)
    colnames(data) <- as.character(barcode)
    colnames(data) <- as.character(paste0(c(1:dim(data)[2]),barcode))
    rownames(data) <- as.character(paste0(c(1:dim(data)[1]),feature))
    return (data)
}

write_h5 <- function(exprs_list, h5file_list) {
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

run_Seurat_RNA_ADT <- function(rna_paths, adt_paths){
  if (length(rna_paths) == 1) {
    rna <- h5_to_matrix(rna_paths[1])
    adt <- h5_to_matrix(adt_paths[1])
    bm <- CreateSeuratObject(counts = rna)
    adt_assay <- CreateAssayObject(counts = adt)
    bm[["ADT"]] <- adt_assay
    
    DefaultAssay(bm) <- 'RNA'
    bm <- NormalizeData(bm) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

    DefaultAssay(bm) <- 'ADT'
    VariableFeatures(bm) <- rownames(bm[["ADT"]])
    bm <- NormalizeData(bm, normalization.method = 'CLR', margin = 2) %>% 
      ScaleData() %>% RunPCA(reduction.name = 'apca')
    
    bm <- FindMultiModalNeighbors(
      bm, reduction.list = list("pca", "apca"), 
      dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
    )

    bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
    bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)
    return(bm)
  } else {
    rna_list <- lapply(rna_paths, function(p) {
      obj <- CreateSeuratObject(counts = h5_to_matrix(p))
      obj <- NormalizeData(obj) %>% FindVariableFeatures()
      return(obj)
    })
    rna_features <- SelectIntegrationFeatures(object.list = rna_list)
    rna_list <- lapply(rna_list, function(x) {
      x <- ScaleData(x, features = rna_features, verbose = FALSE)
      x <- RunPCA(x, features = rna_features, verbose = FALSE)
      return(x)
    })
    rna_anchors <- FindIntegrationAnchors(object.list = rna_list, anchor.features = rna_features, dims = 1:30, reduction = "rpca", reference = 1)
    rna_int <- IntegrateData(anchorset = rna_anchors, dims = 1:30)
    DefaultAssay(rna_int) <- "integrated"
    rna_int <- ScaleData(rna_int) %>% RunPCA(reduction.name = "pca")
    
    adt_list <- lapply(adt_paths, function(p) {
      obj <- CreateSeuratObject(counts = h5_to_matrix(p), assay="ADT")
      obj <- NormalizeData(obj, normalization.method = 'CLR', margin = 2)
      VariableFeatures(obj) <- rownames(obj)
      return(obj)
    })
    adt_features <- SelectIntegrationFeatures(object.list = adt_list)
    adt_list <- lapply(adt_list, function(x) {
      x <- ScaleData(x, features = adt_features, verbose = FALSE)
      x <- RunPCA(x, features = adt_features, verbose = FALSE)
      return(x)
    })
    adt_anchors <- FindIntegrationAnchors(object.list = adt_list, anchor.features = adt_features, dims = 1:18, reduction = "rpca", reference = 1)
    adt_int <- IntegrateData(anchorset = adt_anchors, dims = 1:18)
    DefaultAssay(adt_int) <- "integrated"
    adt_int <- ScaleData(adt_int) %>% RunPCA(reduction.name = "apca")
    
    bm <- rna_int
    bm[["ADT"]] <- adt_int[["integrated"]]
    bm[["apca"]] <- adt_int[["apca"]]
    
    bm <- FindMultiModalNeighbors(
      bm, reduction.list = list("pca", "apca"), 
      dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
    )

    bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
    bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)
    return(bm)
  }
}

run_Seurat_RNA_ATAC <- function(rna_paths, atac_paths){
  if (length(rna_paths) == 1) {
    rna <- h5_to_matrix(rna_paths[1])
    atac <- h5_to_matrix(atac_paths[1])
    pbmc <- CreateSeuratObject(counts = rna)
    atac_assay <- CreateAssayObject(counts = atac)
    pbmc[["ATAC"]] <- atac_assay
    
    DefaultAssay(pbmc) <- "RNA"
    pbmc <- SCTransform(pbmc, verbose = FALSE) %>% RunPCA() %>% RunUMAP(dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')
    
    DefaultAssay(pbmc) <- "ATAC"
    pbmc <- RunTFIDF(pbmc)
    pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
    pbmc <- RunSVD(pbmc)
    pbmc <- RunUMAP(pbmc, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_")
    
    pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50))
    pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
    pbmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, verbose = FALSE)
    return(pbmc)
  } else {
    rna_list <- lapply(rna_paths, function(p) {
      obj <- CreateSeuratObject(counts = h5_to_matrix(p))
      obj <- SCTransform(obj, verbose = FALSE)
      return(obj)
    })
    rna_features <- SelectIntegrationFeatures(object.list = rna_list, nfeatures = 2000)
    rna_list <- PrepSCTIntegration(object.list = rna_list, anchor.features = rna_features)
    rna_list <- lapply(rna_list, function(x) {
      x <- RunPCA(x, features = rna_features, verbose = FALSE)
      return(x)
    })
    rna_anchors <- FindIntegrationAnchors(object.list = rna_list, normalization.method = "SCT", anchor.features = rna_features, dims = 1:50, reduction = "rpca", reference = 1)
    rna_int <- IntegrateData(anchorset = rna_anchors, normalization.method = "SCT", dims = 1:50)
    DefaultAssay(rna_int) <- "integrated"
    rna_int <- RunPCA(rna_int, reduction.name = "pca", npcs = 50)
    
    atac_list <- lapply(atac_paths, function(p) {
      obj <- CreateSeuratObject(counts = h5_to_matrix(p), assay = "ATAC")
      obj <- RunTFIDF(obj) %>% FindTopFeatures(min.cutoff = 'q0') %>% RunSVD()
      return(obj)
    })
    atac_features <- SelectIntegrationFeatures(object.list = atac_list)
    atac_anchors <- FindIntegrationAnchors(object.list = atac_list, anchor.features = atac_features, reduction = "rlsi", dims = 2:50, reference = 1)
    atac_int <- IntegrateData(anchorset = atac_anchors, dims = 2:50, preserve.order = TRUE)
    DefaultAssay(atac_int) <- "integrated"
    atac_int <- ScaleData(atac_int) %>% RunPCA(reduction.name = "lsi", npcs = 50)
    
    bm <- rna_int
    bm[["ATAC"]] <- atac_int[["integrated"]]
    bm[["lsi"]] <- atac_int[["lsi"]]
    
    bm <- FindMultiModalNeighbors(bm, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 1:49))
    bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
    bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, verbose = FALSE)
    return(bm)
  }
}

LT_RNA_ADT_qRNA <- function(reference, query_path){
  DefaultAssay(reference) <- 'RNA'
  query_data <- h5_to_matrix(query_path)
  query_data <- CreateSeuratObject(counts = query_data)
  query_data <- NormalizeData(query_data, verbose = FALSE)
  anchors <- FindTransferAnchors(
    reference = reference,
    query = query_data,
    normalization.method = "LogNormalize",
    reference.reduction = "spca",
    dims = 1:30 #consistent with the WNN methods
  )
  
  query_data <- MapQuery(
    anchorset = anchors,
    query = query_data,
    reference = reference,
    refdata = list(
      celltype.l1 = "celltype",
      predicted_ADT = "ADT"
    ),
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )
  return(query_data)
}

LT_RNA_ATAC_qRNA <- function(reference, query_path){
  query_data <- h5_to_matrix(query_path)
  query_data <- CreateSeuratObject(counts = query_data)
  query_data <- SCTransform(query_data, verbose = FALSE)
  anchors <- FindTransferAnchors(
    reference = reference,
    query = query_data,
    normalization.method = "SCT",
    reference.reduction = "spca",
    dims = 1:50
  )
  
  query_data <- MapQuery(
    anchorset = anchors,
    query = query_data,
    reference = reference,
    refdata = list(
      celltype.l1 = "celltype",
      predicted_ATAC = "ATAC"
    ),
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )
  return(query_data)
}
