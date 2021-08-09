library(Seurat)
library(SeuratDisk)

setwd('./data')
source('https://raw.githubusercontent.com/dosorio/utilities/master/singleCell/scQC.R')

#h5ad to seurat
Convert("hca_heart_vascular_raw.h5ad", dest = "h5seurat", overwrite = TRUE)
seuratObj <- LoadH5Seurat("hca_heart_vascular_raw.h5seurat")

#view counts/data/scale.data
mat <- GetAssayData(object = seuratObj, slot = "data")
df <- as.data.frame(as.matrix(mat)) 

#convert factors to chara
seuratObj[['cell_states']] <- lapply(seuratObj[['cell_states']], as.character)
seuratObj[['cell_type']] <- lapply(seuratObj[['cell_type']], as.character)

seuratObj <- scQC(seuratObj)
seuratObj <- NormalizeData(seuratObj) 
seuratObj <- FindVariableFeatures(seuratObj)

#seurat to h5ad
SaveH5Seurat(seuratObj, filename = "hca_heart_vascular.h5Seurat")
Convert("hca_heart_vascular.h5Seurat", dest = "h5ad", assays = list(SCT = c("counts", "data")))
# Adding data from RNA as X
# Transfering meta.features to var
# Adding counts from RNA as raw
# Transfering meta.features to raw/var

