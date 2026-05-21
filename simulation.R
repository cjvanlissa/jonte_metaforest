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
  studylevel = c(TRUE, FALSE),
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

# prepare parallel processing
nclust <- (parallel::detectCores()-1) ## fixed the number of cores to the HPC limit
cl <- makeCluster(nclust)
doSNOW::registerDoSNOW(cl)

a <- Sys.time()
# run simulation


tab <- foreach(rownum = 1:nrow(summarydata),
               .packages = c("pema", "metafor", "metaforest", "ranger"),
               .combine = rbind) %dopar% {
                 with(as.list(summarydata[rownum, ]), {
                   .Random.seed <- seed[[1]]
                   dat <- simulate_smd(k_train = k_train, k_test = 100, n = n, n2 = 4, es = es, tau2 = tau2, moderators = moderators, model = model, studylevel = studylevel)
                   frm = as.formula(paste0("yi ~ ", paste0("X", 1:moderators, collapse = " + ")))
                   results <- list(
                     mf = metaforest::MetaForest(formula = frm,
                                                 data = dat[dat$training == 1, ]),
                     mf2  = metaforest::MetaForest(formula = frm,
                                                   data = dat[dat$training == 1, ],
                                                   study = "id_exp"),
                     pema = pema::brma(formula = frm,
                                       data = dat[dat$training == 1, ],
                                       study = "id_exp"),
                     mlrf = mlrf(formula = frm,
                                 data = dat[dat$training == 1, ],
                                 study = "id_exp")
                     )

                   preds_train <- list(
                     mf = tryCatch(metaforest:::predict.MetaForest(results$mf)$predictions, error = function(e)NULL),
                     mf2 = tryCatch(metaforest:::predict.MetaForest(results$mf2)$predictions, error = function(e)NULL),
                     pema = tryCatch(pema:::predict.brma(results$pema), error = function(e)NULL),
                     mlrf = tryCatch(predict.mlrf(results$mlrf, newdata = dat[dat$training == 1, ]), error = function(e)NULL)
                   )
                   preds_test <- list(
                     mf = tryCatch(metaforest:::predict.MetaForest(results$mf, data = dat[dat$training == 0, ])$predictions, error = function(e)NULL),
                     mf2 = tryCatch(metaforest:::predict.MetaForest(results$mf2, data = dat[dat$training == 0, ])$predictions, error = function(e)NULL),
                     pema = tryCatch(pema:::predict.brma(results$pema, newdata = dat[dat$training == 0, ]), error = function(e)NULL),
                     mlrf = tryCatch(predict.mlrf(results$mlrf, newdata = dat[dat$training == 0, ]), error = function(e)NULL)
                   )
                   mn_train <- mean(dat$yi[dat$training == 1])
                   r2_train <- unlist(lapply(preds_train, rsq_numeric, obs = dat$yi[dat$training == 1], mn = mn_train))
                   r2_test <- unlist(lapply(preds_test, rsq_numeric, obs = dat$yi[dat$training == 0], mn = mn_train))
                   importance_true <- c(mf = tryCatch(unname(results$mf$forest$variable.importance[1] > 0), error = function(e)NA),
                                        mf2 = tryCatch(unname(results$mf2$forest$variable.importance[1] > 0), error = function(e)NA),
                                        pema = tryCatch(sum(sign(results$pema$coefficients["X1", c("2.5%", "97.5%")])) == 2, error = function(e)NA),
                                        mlrf = tryCatch(unname(results$mlrf$rf$variable.importance[1] > 0), error = function(e)NA))
                   importance_false <- c(mf = tryCatch(unname(tail(results$mf$forest$variable.importance, 1) > 0), error = function(e)NA),
                                         mf2 = tryCatch(unname(tail(results$mf2$forest$variable.importance, 1) > 0), error = function(e)NA),
                                         pema = tryCatch(sum(sign(results$pema$coefficients[nrow(results$pema$coefficients)-2L, c("2.5%", "97.5%")])) == 2, error = function(e)NA),
                                         mlrf = tryCatch(unname(tail(results$mlrf$rf$variable.importance, 1) > 0), error = function(e)NA))
                   c(r2_train, r2_test, importance_true, importance_false)
                 })
               }
b <- Sys.time()
b - a

#Close cluster
stopCluster(cl)
## stop("End of simulation")
saveRDS(tab, "tab.RData")
#save.image(paste0("tab", Sys.Date(), ".RData"))
# load("tab2026-05-18.RData")
colnames(tab)

# Read files --------------------------------------------------------------
library(data.table)

res <- as.data.table(readRDS(list.files(pattern = "summarydata")))
res[, seed := NULL]

design_factors <- names(res)[!names(res) %in% c("rownum", "ndataset")]
res[, (design_factors) := lapply(design_factors, function(thisvar){
  ordered(res[[thisvar]], levels = hyper_parameters[[thisvar]])
})]


colnames(tab) <- paste0(rep( c("r2_train", "r2_test", "importance_true", "importance_false"), each = 4), ".", colnames(tab))

merged <- data.table(res, tab)

fwrite(merged, paste0("sim_results_", Sys.Date(), ".csv"))
saveRDS(merged, paste0("sim_results_", Sys.Date(), ".RData"))

# r2test <- merged[, .SD, .SDcols = grep("r2_test", names(merged), value = TRUE, fixed = TRUE)]
#
# merged[, best := factor(c("mf", "mf2", "pema", "mlrf")[apply(.SD, 1, which.max)]), .SDcols = grep("r2_test", names(merged), value = TRUE, fixed = TRUE)]
#
# data_sum <- merged[ , .(group_sum = table(best)), by = es]
# data_sum
# merged[, best := ]

# Select variables for analysis
varspred <- c("k_train", "es", "tau2", "moderators", "studylevel", "model")

dat <- merged[, .SD, .SDcols = c("ndataset", varspred, grep(".", names(merged), value = TRUE, fixed = TRUE))]

tabres <- melt(dat, measure.vars = grep(".", names(merged), value = TRUE, fixed = TRUE), variable.name = "outcome", value.name = "value")
tabres[, alg := outcome]
levels(tabres$alg) <- gsub("^.+\\.(.+)?$", "\\1", levels(tabres$alg))
levels(tabres$outcome) <- gsub("^(.+)\\..+?$", "\\1", levels(tabres$outcome))

# tabres <- dcast(tabres, es + k + N + p + alg + outcome ~ ., value.var = "value", fun.aggregate = mean)
# setnames(tabres, ".", "correct")

# df_rpart <- tabres[, c("N", "p", "es", "alg", "correct")]
# names(df_rpart) <- c("n", "p", "effect", "algorithm", "correct")

library(rpart)
library(rpart.plot)
library(svglite)

whichbest <- tabres[outcome == "r2_test", .SD, .SDcols = c("ndataset", varspred, "value", "alg")]

whichbest <- dcast(whichbest, as.formula(paste0(paste0(c("ndataset", varspred), collapse = "+"), "~ alg")), value.var = "value")

whichbest[, "best" := factor(levels(tabres$alg)[apply(whichbest[, .SD, .SDcols = levels(tabres$alg)], 1, which.max)])]
# whichbest[, best := factor(best, levels = c("bic", "blrt_05", "srmr"), labels = c("BIC", "BLRT", "PMC"))]

set.seed(783)
fit <- rpart(
  as.formula(paste0("best ~ ", paste0(varspred, collapse = "+"))), data = whichbest, minbucket = round(.01*nrow(whichbest)))

svglite("plot_tree_whichbest.svg", width = 6, height = 4)
rpart.plot(fit)
dev.off()

# write.csv(tabres, "correct_by_cond.csv", row.names = FALSE)
# write.csv(whichbest, "best_algorithm.csv", row.names = FALSE)



# ANOVAs ------------------------------------------------------------------
source("analysis_functions.R")
df_acc <- tabres[outcome == "r2_test", .SD, .SDcols = c("ndataset", varspred, "value", "alg")]
df_acc <- dcast(df_acc, as.formula(paste0(paste0(c("ndataset", varspred), collapse = "+"), "~ alg")), value.var = "value")
setnames(df_acc, levels(tabres$alg), paste0("r2_", levels(tabres$alg)))
names(df_acc) <- gsub(" (CI)", "_ci", names(df_acc), fixed = TRUE)
#the dependent variables (the test r2)
conditions <- varspred
xvars <- conditions
yvars <- setdiff(names(df_acc), c("ndataset", xvars))

#creates a list for every Anova with the results for the anova and the effect sizes for the conditions on the algorithms
anovas<-lapply(yvars, function(yvar){
  form<-paste(yvar, '~', paste(conditions, collapse = "+"))
  # form<-paste(yvar, '~(', paste(unlist(conditions[-lc]), "+", collapse = ' '), conditions[lc], ") ^ 2") #the ^2 signifies that we want all possible effects up until interactions
  thisaov<-aov(as.formula(form), data=df_acc) #change data according to dataframe you are using
  thisetasq<-EtaSq(thisaov)[ , 2]
  list(thisaov, thisetasq)
})

# Anova for the difference ------------------------------------------------

comps <- expand.grid(yvars, yvars)
comps <- comps[!comps$Var1 == comps$Var2, ]
comps <- t(apply(comps, 1, sort))
comps <- comps[!duplicated(comps), ]
#creates a list for every Anova with the results for the anova and the effect sizes for the conditions on the algorithms
diffanovas <- sapply(1:nrow(comps), function(i){
  form<-as.formula(paste("r2", '~ algo * (', paste(conditions, collapse = "+"), ")")) #the ^2 signifies that we want all possible effects up until interactions
  # form<-as.formula(paste("r2", '~ algo * ((', paste(unlist(conditions[-lc]), "+", collapse = ' '), conditions[lc], ") ^ 2)")) #the ^2 signifies that we want all possible effects up until interactions
  tmp <- df_acc
  tmp <- tmp[, .SD, .SDcols = c(conditions, comps[i, , drop = TRUE])]
  names(tmp) <- gsub("^(.+?)_r2", "r2_\\1", names(tmp))
  tmp = melt(tmp, id.vars = conditions,
             measure.vars = names(tmp)[names(tmp) %in% yvars],
             variable.name = "algo",
             value.name = "r2")
  thisaov<-aov(form, data=tmp) #change data according to dataframe you are using
  thisetasq<-EtaSq(thisaov)[ , 2]
  thisetasq <- thisetasq[startsWith(names(thisetasq), "algo")]
  thisetasq
})
colnames(diffanovas) <- paste0(comps[,1], " vs. ", comps[,2])
diffanovas <- data.frame(condition = gsub("algo:", "", rownames(diffanovas), fixed = T), diffanovas)
diffanovas$condition <- trimws(diffanovas$condition)
out <- list(difference = diffanovas[1, ])
#creates dataframe with effect sizes for all conditions for all algorithms
etasqs<-data.frame(sapply(anovas, `[[`, 2))
colnames(etasqs) <- yvars
etasqs$condition <- trimws(rownames(etasqs))
etasqs <- merge(etasqs, diffanovas, by = "condition", all.x = TRUE)
write.csv(etasqs, "effect_of_conditions.csv", row.names = FALSE)


library(ggplot2)
plts <- lapply(varspred, function(cond){
  df_plot <- tabres[outcome == "r2_test", .SD, .SDcols = c("ndataset", varspred, "value", "alg")]
  df_plot <- df_plot[, .(mn = mean(value, na.rm = TRUE)), by = c(cond, "alg")]
  ggplot(df_plot, aes(x = .data[[cond]], y = mn, linetype = alg, group = alg)) +
    geom_line() +
    theme_bw() +
    theme(legend.position = c(.85,.2),
          axis.title.y = element_blank())

})
plts[2:length(plts)] <- lapply(plts[2:length(plts)], function(p){p + theme(legend.position = "none")})
library(ggpubr)

p <- ggarrange(plotlist = plts, ncol = 2, nrow = 3)
ggsave("conditions.svg", p, width = 210, height = 297, units = "mm")
