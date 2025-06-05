## 
## gp_nls_scripts_mod.R
## Project: Sample Size Determination via Learning Curves for AI Models: 
##          An application to Deep Learning Algorithms for Diagnostic and Prediction Tasks
## Authors: The Società Italiana di Statistica Medica ed Epidemiologia Clinica
##          Research Group on Machine Learning in Clinical Research - 
##          subgroup on non-tabular data
## Purpose: Scripts for learning curve fitting adapted from:
##          - Dayimu et al., 2024, doi: 10.1002/sim.10121 (script hosted in
##            https://github.com/adayim/samplesize-learning)
##          - Figueroa et al., 2012, doi: 10.1186/1472-6947-12-8 (script in 
##            Electronic supplementary materials)
## Additional information: a complete list of changes from the original script is provided in changes.txt
## Creation date: 22 March 2024
## Last modification: 4 December 2024
## 


# GP functions adapted from Dayimu et al., 2024 ---------------------------

# Build gp model for logistic regression
gp_model <- function(dat, auc_var = "auc_cros", est = NULL,
                     n.chains=1,n.iter=20000,n.burnin=10000,
                     n.thin=10){#Modified to allow specification of chains, iterations, burnin and thinning
  # dat: dataset contains sample size `n` and corresponding C-statistics `auc_var`
  # auc_var: variable name of contains C-statistics
  # est: estimated posterior summary of external data
  
  require(R2jags, quietly = TRUE)
  
  dat <- na.omit(dat)
  
  dat$auc <- dat[[auc_var]]
  prams <- c("alpha", "beta", "gamma", "sigma_g", "phi", "sigma_y")
  
  if(is.null(est)){
    dat_lst <- list(x = dat$n,
                    y = dat$auc,
                    d = dist(dat$n) |>  as.matrix(),
                    N = nrow(dat),
                    w = dat$w)#n/max(dat$n)) # Modified to allow specification of weights 
    mod_file <- "scripts/gp-mod_trunc.txt"  #modification of original gp-mod.txt with truncation of phi below at value 0.01
  }else {
    # With priori
    # Moment matching
    est_beta_param <- function(val) {
      mu <- val[1]
      var <- val[2]
      alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
      beta <- alpha * (1 / mu - 1)
      c(alpha, beta)
    }
    
    est <- cbind(est, var = est[,"sd"]^2)
    # est <- cbind(est, var = 1) # Set all variance to 1
    est <- est[c("beta", "gamma", "phi", "sigma_g"), c("mean", "var")]
    
    dat_lst <- list(x = dat$n,
                    y = dat$auc,
                    d = dist(dat$n) |>  as.matrix(),
                    N = nrow(dat),
                    w = dat$w,#n/max(dat$n)) # Modified to allow specification of weights 
                    # a = c(1/2, 1/2),
                    b = c(est["beta", 1], 1/est["beta", 2]),
                    # b = c(est["beta", 1], 1),
                    c = est_beta_param(est["gamma", ]),
                    s = c(est["sigma_g", ][1], 1/est["sigma_g", ][2]),
                    # s = c(est["sigma_g", ][1], 1),
                    p = c(est["phi", ][1], 1/est["phi", ][2]))
    
    mod_file <- "scripts/gp-mod-prior.txt"
  }
  
  jags(
    data = dat_lst,
    parameters.to.save = prams,
    model.file = mod_file,
    n.chains = n.chains, # modified to allow specify these parameters
    n.iter = n.iter, 
    n.burnin = n.burnin,
    n.thin = n.thin,
    quiet = TRUE
  )
}

# Summarise prediction and data
predict_gp <- function(fit, dat, auc_var = "auc_cros",x_star=c(seq(50, 2000, by = 10))){#modified so x_star can be specified by the user
  # fit: fitted GP model
  # dat: observed dataset contains sample size `n` and corresponding C-statistics `auc_var`
  # auc_var: variable name of contains C-statistics
  
  dat$auc <- dat[[auc_var]]
  
  dat <- dat[,.(auc=mean(auc, na.rm = TRUE)), .(n)]
  dfm <- data.frame(x = dat$n,
                    y = dat$auc)
  
  x_star <- x_star #modified in order to specify x_star
  pred_mcmc(fit, x_star = x_star, dfm)  |> 
    as_tibble() |> 
    rename(n = x) 
}


# Predict
pred_mcmc <- function(mod, x_star, dat){
  
  alpha <- mod$BUGSoutput$summary["alpha", "mean"]
  beta <- mod$BUGSoutput$summary["beta", "mean"]
  gamma <- mod$BUGSoutput$summary["gamma", "mean"]
  sigma_y <- mod$BUGSoutput$summary["sigma_y", "mean"]
  sigma_g <- mod$BUGSoutput$summary["sigma_g", "mean"]
  phi <- mod$BUGSoutput$summary["phi", "mean"]
  
  x <- dat$x
  y <- dat$y
  
  n_obs <- length(x)
  
  mu <- 1 - alpha - beta * x^(-gamma)
  mu_pred <- 1 - alpha - beta * x_star^(-gamma)
  
  Sigma <-  sigma_y*2 * diag(length(x)) + sigma_g^2*exp(-(phi^2)*fields::rdist(x,x)^2)
  Sigma_star <- sigma_g^2*exp(-(phi^2)*fields::rdist(x_star,x)^2)
  Sigma_star_star <- sigma_g^2*exp(-(phi^2)*fields::rdist(x_star,x_star)^2)
  
  pred_mean <- mu_pred + Sigma_star %*% solve(Sigma) %*% (y - mu)
  pred_var <- Sigma_star_star - Sigma_star %*% solve(Sigma) %*% t(Sigma_star)
  
  data.frame(x = x_star, 
             fit = pred_mean,
             lwr = pred_mean - 1.96 * sqrt(diag(pred_var)), 
             upr = pred_mean + 1.96 * sqrt(diag(pred_var)))
}


# NLS functions adapted from Dayimu et al., 2024 --------------------------

# Fit NSL model
nls_model <- function(dat, auc_var = "auc_cros",start) { #modified in order to provide starting values
  # dat: dataset contains sample size `n` and corresponding C-statistics `auc_var`
  # auc_var: variable name of contains C-statistics
  
  require(boot, quietly = TRUE)
  require(gslnls, quietly = TRUE)
  
  dat <- na.omit(dat)
  dat$auc <- dat[[auc_var]]
  
  # Calculate weight
  if(!"w" %in% names(dat))
    dat$w <- dat$n / max(dat$n)
  
  # Fit model
  gsl_nls(
    fn = auc ~ (1 - inv.logit(alpha)) - (beta * (n^(-inv.logit(gamma)))),
    data = dat,
    start = start,#modified to provide starting values
    algorithm = "lm",
    control = gsl_nls_control(maxiter = 1000000),
    weights = w
  )
  
}

## Prediction ==============
# Confidence/prediction bands for nonlinear regression 
# (i.e., objects of class nls) are based on a linear approximation
# as described in Bates & Watts (2007). Also known as Delta method.
predict_nls <- function(fit, dat, auc_var = "auc_cros",new_data=data.frame(n = seq(50, 2000, by = 10))){#Added new_data to specify predicted points
  # fit: fitted NLS model
  # dat: observed dataset contains sample size `n` and corresponding C-statistics `auc_var`
  # auc_var: variable name of contains C-statistics
  
  dat$auc <- dat[[auc_var]]
  
  #new_data <- data.frame(n = seq(50, 2000, by = 10)) #specified as argument so that it can be specified by the user 
  
  predict(fit, newdata = new_data, interval = "prediction", level= 0.95)  %>% 
    as_tibble() %>% 
    mutate(n = new_data$n)
}


# NLS script adapted from Figueroa et al., 2012 ---------------------------

##################################################################
##################################################################
#need to set the following parameters
#functions to calculate rmse and mae should be defined by the user
#offset = offset to start fitting, it's always set to zero
#i = number of points to include in training data
#W = weights
#N = Total number of points
#startParams = start parameters to nls function currently set to (a=0,b=1,c=-0.5)
FitModel<- function(offset, X, Y, W, i, N,startParams)
{
  #Data considering only points between offset and i
  x<-X[offset:i];
  y<-Y[offset:i];
  w<-W[offset:i];
  gradientF<-deriv3(~(1-a)-(b*(x^c)), c("a","b","c"), function(a,b,c,x) NULL);
  # fitting the model using nls
  m<-nls(y~gradientF(a,b,c,x), start = startParams, weights=w,
         control = list(maxiter=1000, warnOnly = TRUE),
         algorithm = "port", upper = list(a=10, b = 30, c = -0.1),#Modified upper bound
         lower = list(a = 0, b = 0, c=-10), data = data.frame(y=y, x=x))
  print(m) #Added to print the fitted parameters
  #predict Y for sample sizes not used to fit the curve
  #if all data was used to fit model, testing data = training data
  #else, testing data = (total data - training data)
  if (i==N){
    testX<-X[(offset:i)];
    testY<-Y[(offset:i)];
    testW <- W[offset:i];
  }
  else{
    testX<-X[((i+1):N)]; #Get remaining X
    testY<-Y[((i+1):N)];
    testW<-W[((i+1):N)];
  }
  #predictions on unseen data
  prediction<-predict(m, list(x=testX));
  #confidence intervals
  se.fit <- sqrt(apply(attr(predict(m, list(x=testX)),"gradient"),1,
                       function(x) sum(vcov(m)*outer(x,x))));
  prediction.ci <- prediction + outer(se.fit,qnorm(c(.5, .025,.975)));
  predictY<-prediction.ci[,1];
  predictY.lw<-prediction.ci[,2];
  predictY.up<-prediction.ci[,3];
  #Calculate residuals
  if (i==N) res<-rep(0,length(X))
  else res<-rep(0,length(X-i))
  res<-(predictY-testY);
  #Calculate Root Mean square error
  rmseValue<-rmse(testY, predictY);
  #Calculate Absolute error
  maeValue<-abse(testY, predictY);
  #Added to return the predicted values
  results<-list(prediction_db=data.frame(n=X[(i+1):N],predicted=predictY,
                                         predicted.lb=predictY.lw,
                                         predicted.ub=predictY.up),
                residuals=res,
                RMSE=rmseValue,
                ABSE=maeValue,
                parms=m$m$getPars())
  return(results)
}
