mf_sim <- function(cl, simulated_data, file_stem, ...){
  mf_r_models <- parLapply(cl, simulated_data, function(data) {
    MetaForest(yi ~.,
               data = data$training,
               ...)
  })
  
  mf_r_fits <- t(
    clusterMap(
      cl,
      fun = function(models, data) {
        c(
          model_accuracy(
            fit = models,
            newdata = subset(data[[1]], select = -c(1, 2)),
            observed = data[[1]]$yi
          ),
          model_accuracy(
            fit = models,
            newdata = subset(data[[2]], select = -1),
            observed = data[[2]]$yi,
            ymean = mean(data[[1]]$yi)
          ),
          tau2 = models$rma_after$tau2
        )
      },
      models = mf_r_models,
      data = simulated_data,
      SIMPLIFY = TRUE,
      USE.NAMES = FALSE
    )
  )
  
  mf_r_importance <- parLapply(
    cl = cl,
    mf_r_models,
    fun = function(models) {
      models$forest$variable.importance
    }
  )
  
  file_name <-
    paste0(paste(c(file_stem, "fits", chunk), collapse = "_"), ".RData")
  saveRDS(mf_r_fits, file = file_name, compress = FALSE)
  
  file_name <-
    paste0(paste(c(file_stem, "selected", chunk), collapse = "_"), ".RData")
  saveRDS(mf_r_importance,
          file = file_name,
          compress = FALSE)
  
  rm(mf_r_fits, mf_r_importance, mf_r_models)
}
