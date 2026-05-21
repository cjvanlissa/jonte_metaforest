rsq_numeric <- function(obs, preds, mn){
  if(is.null(preds)) return(NA)
  tss <- sum((obs-mn)^2)
  rss <- sum((preds - obs) ^ 2)
  return(1 - rss/tss)
}
