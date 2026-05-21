mlrf <- function (formula, data, vi = "vi", study = NULL, whichweights = "random", num.trees = 500,
          mtry = NULL, min.node.size = 5, method = "REML", random = ~ 1 | id_exp/id_es, ...)
{
  v <- data[[vi]]
  id <- data[[study]]
  mf <- model.frame(formula = formula, data = data)
  yi <- mf[[1]]
  X <- model.matrix(formula, data)[, -1, drop = FALSE]
  if(study %in% colnames(X)) X <- X[, !colnames(X) == study]
  if(vi %in% colnames(X)) X <- X[, !colnames(X) == vi]
  # Your exact 3-level meta-analytic model
  m3 <- rma.mv(
    yi = yi,
    V  = vi,
    random = random, #~ 1 | id_exp/ES_ID,
    data = data,
    method = "REML"
  )

  resid <- residuals(m3, type = "response")
  wts <- sqrt(1/v)

  rf_mod <- ranger::ranger(
    x = X,
    y = yi,
    case.weights = wts,
    num.trees = num.trees,
    importance = "permutation")

  out <- list(m3 = m3, rf = rf_mod)
  class(out) <- c("mlrf", class(out))
  return(out)
}

predict.mlrf <- function(object, newdata = NULL, ...){
  predict(object[["m3"]])$pred + ranger:::predict.ranger(object[["rf"]], data = newdata, type = "response")$predictions
}
