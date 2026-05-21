simulate_smd <- function(k_train = 20, k_test = 100, n = 40, n2 = 4, es = .5,
                         tau2 = 0.04, moderators = 5, model = "es * x[, 1]", studylevel = FALSE)
{
  # Make label for training and test datasets
  training <- c(rep(1, k_train*n2), rep(0, k_test*n2))
  id_exp <- c(rep(1:k_train, each = n2), rep(k_train + c(1:k_test), each = n2))
  n = rep(n, length(training))

  # Generate moderator matrix x:
  if(studylevel){
    x <- matrix(sample.int(2, size = (k_train+k_test) * moderators, replace = TRUE)-1L, ncol = moderators)
    x <- x[rep(1:nrow(x), each = n2), ]
  } else {
    x <- matrix(sample.int(2, size = length(n) * moderators, replace = TRUE)-1L, ncol = moderators)
  }
  # Sample true effect sizes theta.i from a normal distribution with mean
  # mu, and variance tau2, where mu is the average
  # population effect size. The value of mu depends on the values of the
  # moderators and the true model mu <- eval(model)
  model <- parse(text = model)
  mu <- eval(model)

  # theta.i: true effect size of study i
  theta.i <- rep(rnorm(n = (k_train+k_test), sd = sqrt(tau2)), each = n2) + mu

  # Then the observed effect size yi is sampled from a non-central
  # t-distribution under the assumption that the treatment group and
  # control group are both the same size
  p_ntk <- 0.5  #Percentage of cases in the treatment group
  ntk <- p_ntk * n  #n in the treatment group for study i
  nck <- (1 - p_ntk) * n  #n in the control group for study i
  df <- n - 2  #degrees of freedom
  j <- 1 - 3/(4 * df - 1)  #correction for bias
  nk <- (ntk * nck)/(ntk + nck)
  ncp <- theta.i * sqrt(nk)  #Non-centrality parameter

  # Standardized mean difference drawn from a non-central t-distribution
  SMD <- mapply(FUN = rt, n = 1, df = df, ncp = ncp)

  # yi is Hedges' g for study i
  yi <- SMD/((j^-1) * (nk^0.5))

  # Calculate the variance of the effect size
  vi <- j^2 * (((ntk + nck)/(ntk * nck)) + ((yi/j)^2/(2 * (ntk + nck))))

  # Dersimonian and Laird estimate of tau2
  Wi <- 1/vi[1:k_train]
  tau2_est <- max(0, (sum(Wi * (yi[1:k_train] - (sum(Wi * yi[1:k_train])/sum(Wi)))^2) -
                        (k_train - 1))/(sum(Wi) - (sum(Wi^2)/sum(Wi))))

  return(data.frame(training, id_exp, id_es = 1:length(training), vi, yi, x))

  # list(training = subset(data, training == 1, -1), testing = subset(data,
  #                                                                   training == 0, -c(1, 2)), housekeeping = data.frame(n = n, mu_i = mu, theta_i = theta.i),
  #      tau2_est = tau2_est)
}

n_to_vi <- function(n){
  3.84417/n^.93728
}
