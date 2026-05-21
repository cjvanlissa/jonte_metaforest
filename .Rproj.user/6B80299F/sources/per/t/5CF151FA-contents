#clear console and env
rm(list=ls(all.names = T))
cat("\014")

#load necessary packages
# Install packages; skip if already installed -----------------------------
# install.packages("metaforest")
# install.packages("pema")
# install.packages("foreach") # For parallel computing
# install.packages("sn")

# Load packages
library(parallel) # For parallel computing
library(sn)
library(metaforest) # Contains the function to simulate datasets
library(metafor)

library(foreach)
library(data.table)


# Load simulation functions from source -----------------------------------
source('sim_functions.R')
source('simulate_smd.R')
source('mlrf.R')

# set conditions for simulation
hyper_parameters <- list(
  #Number of datasets per condition
  ndataset=1:100,
  #Number of studies per dataset, normally distributed with mean n and sd n/3
  k_train=c(20, 40, 80, 120),
  #Average n per study (k)
  n=c(100),
  #Effect size
  es=c(0, .2, .5, .8),
  #Residual heterogeneity
  tau2=c(0, 0.01, .04, .28),
  # # Slant parameter alpha
  # alpha_tau = c(0),
  # alpha_mod = c(0, 2, 10),
  #Study-level moderators
  moderators= c(2, 5),
  model = c("es * x[, 1]", "es * x[, 1] + es * x[, 2] + es * (x[, 1] * x[, 2])")
)

# Create hypergrid with simulation parameters and save it as .RData file extension
summarydata <- expand.grid(hyper_parameters, stringsAsFactors = FALSE)
summarydata$rownum <- 1:nrow(summarydata)
# Remove unnecessary 0 effects
summarydata <- summarydata[!(summarydata$es == 0 & !summarydata$model == "es * x[, 1]"), ]

# Prepare seeds
RNGkind("L'Ecuyer-CMRG") # 1. RNG capable of creating multiple streams
set.seed(70261)
seeds <- vector("list", nrow(summarydata))
seeds[[1]] <- .Random.seed
for (i in 2:length(seeds)) {
  seeds[[i]] <- parallel::nextRNGStream(seeds[[i - 1]])
}
summarydata$seed <- seeds

saveRDS(summarydata, file = "summarydata.RData")

#if(dir.exists("results")){
#  unlink("results", recursive = T)
#}
#dir.create("results")

# prepare parallel processing
nclust <- 10 #(parallel::detectCores()-1) ## fixed the number of cores to the HPC limit
cl <- makeCluster(nclust)
doSNOW::registerDoSNOW(cl)

a <- Sys.time()
# run simulation

tab <- foreach(rownum = 1:nclust, #nrow(summarydata),
               .packages = c("pema", "metafor", "metaforest", "ranger"),
               #.export = c("rsq_numeric", "simulate_smd"),
               .combine = rbind) %dopar% {

  # Set seed
  tryCatch({
    attach(summarydata[rownum, ])
    .Random.seed <- seed[[1]]
    dat <- simulate_smd(k_train = k_train, k_test = 100, n = n, n2 = 4, es = es, tau2 = tau2, moderators = moderators, model = model)
    frm = as.formula(paste0("yi ~ ", paste0("X", 1:moderators, collapse = " + ")))
    results <- list(
      mf = tryCatch({
        metaforest::MetaForest(formula = frm,
                               data = dat[dat$training == 1, ])}, error = function(e){NULL}),
      mf2  = tryCatch({
        metaforest::MetaForest(formula = frm,
                               data = dat[dat$training == 1, ],
                               study = "id_exp")}, error = function(e){NULL}),
      pema =  tryCatch({
        pema::brma(formula = frm,
                   data = dat[dat$training == 1, ],
                   study = "id_exp")}, error = function(e){NULL}),
      mlrf = tryCatch({
        mlrf(formula = frm,
             data = dat[dat$training == 1, ],
             study = "id_exp")}, error = function(e){NULL})
    )

    preds_train <- list(
      mf = tryCatch(metaforest:::predict.MetaForest(results$mf)$predictions, error = function(e){NULL}),
      mf2 = tryCatch(metaforest:::predict.MetaForest(results$mf2)$predictions, error = function(e){NULL}),
      pema = tryCatch(pema:::predict.brma(results$pema), error = function(e){NULL}),
      mlrf = tryCatch(predict(results$mlrf, newdata = dat[dat$training == 1, ]), error = function(e){NULL})
    )
    preds_test <- list(
      mf = tryCatch(metaforest:::predict.MetaForest(results$mf, data = dat[dat$training == 0, ])$predictions, error = function(e){NULL}),
      mf2 = tryCatch(metaforest:::predict.MetaForest(results$mf2, data = dat[dat$training == 0, ])$predictions, error = function(e){NULL}),
      pema = tryCatch(pema:::predict.brma(results$pema, newdata = dat[dat$training == 0, ]), error = function(e){NULL}),
      mlrf = tryCatch(predict(results$mlrf, newdata = dat[dat$training == 0, ]), error = function(e){NULL})
    )
    mn_train <- mean(dat$yi[dat$training == 1])
    r2_train <- unlist(lapply(preds_train, rsq_numeric, obs = dat$yi[dat$training == 1], mn = mn_train))
    r2_test <- unlist(lapply(preds_test, rsq_numeric, obs = dat$yi[dat$training == 0], mn = mn_train))
    importance_true <- c(mf = unname(results$mf$forest$variable.importance[1] > 0),
                         mf2 = unname(results$mf2$forest$variable.importance[1] > 0),
                         pema = sum(sign(results$pema$coefficients["X1", c("2.5%", "97.5%")])) == 2,
                         mlrf = unname(results$mlrf$rf$variable.importance[1] > 0))
    importance_false <- c(mf = unname(tail(results$mf$forest$variable.importance, 1) > 0),
                          mf2 = unname(tail(results$mf2$forest$variable.importance, 1) > 0),
                          pema = sum(sign(results$pema$coefficients[nrow(results$pema$coefficients)-2L, c("2.5%", "97.5%")])) == 2,
                          mlrf = unname(tail(results$mlrf$rf$variable.importance, 1) > 0))
    c(r2_train, r2_test, importance_true, importance_false)
  }, error = function(e){rep(NA, 16)})

}
b <- Sys.time()
b - a

#Close cluster
stopCluster(cl)
## stop("End of simulation")

save.image(paste0("tab", Sys.Date(), ".RData"))


# Read files --------------------------------------------------------------
library(data.table)

res <- as.data.table(readRDS(list.files(pattern = "summarydata")))
res[, seed := NULL]
res[, reps := NULL]
design_factors <- names(res)[!names(res) %in% c("rownum", "reps")]
res[, (design_factors) := lapply(design_factors, function(thisvar){
  ordered(res[[thisvar]], levels = hyper_parameters[[thisvar]])
})]

max_class <- 6
reps <- 2

# Rename
vars <- c("rownum", "rep",
          paste0("M_", paste0(c("mm", "srmr"), "_", rep(1:max_class, each = reps))), # rowMeans(rep_stat)
          paste0("SD_", paste0(c("mm", "srmr"), "_", rep(1:max_class, each = reps))), # apply(rep_stat, 1, sd)
          paste0(rep(c("lb", "ub"), 6), "_", paste0(rep(c("mm", "srmr"), each = reps), "_", rep(1:max_class, each = 4))), # as.vector(apply(rep_stat, 1, quantile, probs = c(.025, .975)))
          paste0("srmr_.1_", 1:max_class), # rowMeans(rep_stat[seq(from = 2, to = nrow(rep_stat), by = 2), ] < .1)
          paste0("mm_min_dat_", 1:max_class), # rowMeans(rep_stat[seq(from = 1, to = nrow(rep_stat), by = 2), ] - mr_dat)
          paste0("mm_greater_dat_", 1:max_class), # rowMeans(rep_stat[seq(from = 1, to = nrow(rep_stat), by = 2), ] > mr_dat)
          paste0("dif_srmr_", 1:(max_class-1), "_", 2:max_class), # rowMeans(dif_sr)
          paste0("dif_srmr_", rep(c("lb", "ub"), 2), "_", paste0(rep(1:(max_class-1), each = reps), "_", rep(2:max_class, each = reps))), # as.vector(apply(dif_sr, 1, quantile, probs = c(.025, .975)))
          paste0("dif_srmr_neg_", 1:(max_class-1), "_", 2:max_class), # rowMeans(dif_sr < 0)
          paste0("dif_mm_", 1:(max_class-1), "_", 2:max_class), # rowMeans(dif_mr)
          paste0("dif_mm_", rep(c("lb", "ub"), 2), "_", paste0(rep(1:(max_class-1), each = reps), "_", rep(2:max_class, each = reps))), # as.vector(apply(dif_mr, 1, quantile, probs = c(.025, .975)))
          paste0("dif_mm_neg_", 1:(max_class-1), "_", 2:max_class),# rowMeans(dif_mr < 0)
          paste0("blrt_pv_", 1:(max_class-1), "_", 2:max_class), #blrt$p.value,
          paste0("AICS_", 1:max_class), #IC$aics,
          paste0("BICS_", 1:max_class), #IC$bics,
          #"AIC_min", #which.min(IC$aics),
          #"BIC_min", #which.min(IC$bics),
          paste0("AIC_DCI_", 1:(max_class-1), "_", 2:max_class),#ic_ci[,"AIC_0"],
          paste0("BIC_DCI_", 1:(max_class-1), "_", 2:max_class)#ic_ci[,"BIC_0"]
          )

#f <- list.files("results", full.names = TRUE)
#tab <- lapply(f, fread, header = F)
#tab <- rbindlist(tab)
#setorderv(tab, cols = c("V1"), order=1L, na.last=FALSE)
#if(!(tab$V1[1] == 1 & tail(tab$V1, 1) == nrow(res) & length(unique(tab$V1)) == nrow(res))){
#  c(1:nrow(res))[!c(1:nrow(res)) %in% unique(tab$V1)]
#  stop()
#}
colnames(tab) <- vars
#tab[, "rownum" := NULL]
merged <- merge(res, tab, by = "rownum")

fwrite(merged, paste0("sim_results_", Sys.Date(), ".csv"))
saveRDS(merged, paste0("sim_results_", Sys.Date(), ".RData"))
