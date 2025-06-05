##
## ML_sample_size_main.R
## Project: Sample Size Determination via Learning Curves for AI Models: An
##          application to Deep Learning Algorithms for Diagnostic and
##          Prediction Tasks
## Authors: The Società Italiana di Statistica Medica ed Epidemiologia Clinica
##          Research Group on Machine Learning in Clinical Research - 
##          subgroup on non-tabular data
## Purpose: Main script - Fitting learning curves to predict target AUCs in a 
##          classification and in a prediction task
## Creation date: 22 March 2024
## Last modification: 4 December 2024
## System dependencies: JAGS (installed version: 4.3.1)
##

library(R2jags)
library(data.table) # Required by Dayimu et al., 2021 script
library(gslnls) #Required by Dayimu et al., 2021 script
library(boot) # Required by Dayimu et al., 2021 script
library(dplyr)
library(Metrics)
library(tidyverse)

source('scripts/gp_nls_scripts.R')

# Function for retrieving n at which specific thresholds are reached by the predicted value
find_n_threshold <- function(dat_predicted_values, var, thresholds){
  return_dat<-data.frame(n=integer(),fit=double(),lwr=character(),upr=character())
  for(i in 1:length(thresholds)){
    return_dat[i,]<-dat_predicted_values %>%
      arrange(n) %>%
      filter(var >= thresholds[i]) %>%
      slice(1)
  }
  return_dat$threshold<-thresholds
  return(return_dat)
}
rmse<-Metrics::rmse # Specification of RMSE and ABSE function required by Dayimu et al. script
abse<-Metrics::mse

# CLASSIFICATION TASK ----------------------------------------------------

#Load training points
db<-read.csv("data/AFdetection_sample1000_100reps.csv")
db = data.frame(t(db))
auc_var =rowMeans(db[1:100])
n = seq(100, 900, by=50)
se<-apply(db[1:100],1,sd)/sqrt(100)
db$n = n
db$auc_var = auc_var
db$se <- se
db1<-select(db, c(n,auc_var,se))
dat_class<-data.table(db1)

#Load test points
test_class<-read.csv("data/AFdetection_sample80254_testPoints.csv")
db_test_class=data.frame(t(test_class))[2:5,]
auc_test_class<-rowMeans(db_test_class,na.rm = TRUE)
n_test_class <- c(2000,5000,10000,70000)
sd_test_class<-apply(db_test_class[1:10],1,sd,na.rm=TRUE)
nrep_class<-rowSums(!is.na(db_test_class))
se_test_class<-sd_test_class/sqrt(nrep_class)
dat_test_class<-data.frame(n=n_test_class,auc_var=auc_test_class,se=se_test_class)

# Using Figueroa, 2012 functions -----------------------------------------------------
start<-c(a=0.01,b=1,c=-0.5) # starting values that can be used also with Dayimu et al. functions  
offset<-0 

#This model is run to check whether the nls method works well
i<-11
dat_check<-dat_class
dat_check$w<-ifelse(row(dat_class[,1])<=i,1:i/i,NA) #Weights based on cardinality

res_nls_class_check<-FitModel(offset=offset,dat_check$n,dat_check$auc_var,
                              dat_check$w,i = i,N=nrow(dat_check),
                              start=start) #b would have reached upper bound if original script was used (previous upper bound for b = 10)
res_nls_class_check 

dat_plot_check<-merge(dat_check,res_nls_class_check$prediction_db,by.x="n",by.y="n",all=TRUE) #Dataset with observed and predicted values
plot(dat_plot_check$n,dat_plot_check$auc_var,ylim=c(0.5,1),col="green",bg="green",pch=21)
points(dat_plot_check$n,dat_plot_check$predicted,col="red",bg="red",pch=21)
lines(dat_plot_check$n,dat_plot_check$predicted.lb)
lines(dat_plot_check$n,dat_plot_check$predicted.ub)
polygon(c(dat_plot_check$n,rev(dat_plot_check$n)),
        c(dat_plot_check$predicted.lb,rev(dat_plot_check$predicted.ub)),
        col="#ee660050",border=NA)
text(x=c(550,700),y=0.55,labels=c("RMSE:",round(res_nls_class_check$RMSE,3)))


# Predict beyond the training points and check against test points
dat_class_v2<-merge(dat_class,dat_test_class,by=c("n","auc_var","se"),all=TRUE)
i2<-17
dat_class_v2$w<-ifelse(row(dat_class_v2[,1])<=i2,1:i2/i2,NA)

res_nls_class_fig<-FitModel(offset=offset,dat_class_v2$n,dat_class_v2$auc_var,
                        dat_class_v2$w,i = i2,N=nrow(dat_class_v2),start=start)
res_nls_class_fig

# Visualize predicted and actual test AUC 
dat_plot_class<-merge(dat_class,res_nls_class_fig$prediction_db,by.x="n",by.y="n",all=TRUE)
plot(dat_plot_class$n,dat_plot_class$auc_var,ylim=c(0.5,1))
lines(dat_plot_class$n,dat_plot_class$predicted,lty = 2)
points(dat_plot_class$n,dat_plot_class$predicted,col="red",bg="red",pch=21)
lines(dat_plot_class$n,dat_plot_class$predicted.lb)
lines(dat_plot_class$n,dat_plot_class$predicted.ub)
polygon(c(dat_plot_class$n,rev(dat_plot_class$n)),
        c(dat_plot_class$predicted.lb,rev(dat_plot_class$predicted.ub)),
        col="#ee660050",border=NA)
points(dat_test_class$n,dat_test_class$auc,pch=21,bg="green")
text(x=c(50000),y=0.6,labels=paste0("RMSE: ",round(res_nls_class_fig$RMSE,3)))

# Compute n where pre-specified AUCs are reached
extra_data_class<-data.frame(n=seq(1000,70000,500),auc_var=NA)
dat_class_extra<-merge(dat_class,extra_data_class,all=TRUE)
dat_class_extra$w<-ifelse(row(dat_class_extra[,1])<=i2,1:i2/i2,NA) #Weights based on cardinality

res_nls_class_extra<-FitModel(offset=offset,dat_class_extra$n,dat_class_extra$auc_var,
                              dat_class_extra$w,i = i2,N=nrow(dat_class_extra),start=start) 

aucs_class_nls_fig<-find_n_threshold(res_nls_class_extra$prediction_db,
                                     round(res_nls_class_extra$prediction_db$predicted,3),
                                     thresholds = c(.95,.97,.99))
  
aucs_class_nls_fig$method<-"nls"
aucs_class_nls_fig$alg<-"Figueroa et al., 2012"

# Using Dayimu, 2021 functions -------------------------------------------------------
dat_class_v3<-dat_class
dat_class_v3$w<-ifelse(row(dat_class_v3[,1])<=nrow(dat_class_v3),1:nrow(dat_class_v3)/nrow(dat_class_v3),NA)

# Nonlinear least squares
# Since Dayimu's implementation use a logit transformation of a and c, and computes c as a positive value
# (the sign is modified after estimating c), we specify here a = logit(0.01), b same as Figueroa, and c = logit(0.5) 
res_nls_class_day<-nls_model(dat = dat_class_v3,auc_var = "auc_var",
                            start=list(alpha = -4.5951199, beta = 1, gamma = 0))
res_nls_class_day

inv.logit(-16.013)
inv.logit(1.673)#Values similar to those obtained through Figueroa's script

preds_class_nls<-predict_nls(fit=res_nls_class_day,dat=dat_class_v3,
                             auc_var = "auc_var",
                             new_data=data.frame(n = seq(1000, 70000, by = 500)))%>%
  relocate(n,.before=fit)
preds_class_nls

# Compute the RMSE
dat_comp_nls_class<-preds_class_nls[preds_class_nls$n==2000|preds_class_nls$n==5000|
                                      preds_class_nls$n==10000|preds_class_nls$n==70000,]
rmse(dat_test_class$auc,dat_comp_nls_class$fit)

# n to reach target AUCs
aucs_class_nls_day<-find_n_threshold(preds_class_nls,
                                     round(preds_class_nls$fit,3),
                                     thresholds = c(.95,.97,.99))
aucs_class_nls_day$alg<-"Dayimu et al., 2024"
aucs_class_nls_day$method<-"nls"

# Gaussian process
set.seed(123)
res_gp_class<-gp_model(dat = dat_class_v3,auc_var = "auc_var",n.chains=5,
                       n.iter=2000000,n.burnin=300000,n.thin=10)#set the same number of iteration that will be use for predictive task
res_gp_class
# Stability of results at different seeds are assessed in the "additional analyses" script 

preds_class_gp<-predict_gp(fit=res_gp_class,dat=dat_class_v3,auc_var = "auc_var",x_star <- seq(1000, 70000, by = 500))

dat_comp_gp_class<-preds_class_gp[preds_class_gp$n==2000|preds_class_gp$n==5000|
                                    preds_class_gp$n==10000|preds_class_gp$n==70000,]
rmse(dat_test_class$auc_var,dat_comp_gp_class$fit)


plot(dat_class_v3$n,dat_class_v3$auc_var,ylim=c(0.5,1),xlim=c(0,70000))
lines(preds_class_gp$n,preds_class_gp$lwr)
lines(preds_class_gp$n,preds_class_gp$upr)
polygon(c(preds_class_gp$n,rev(preds_class_gp$n)),
        c(preds_class_gp$lwr,rev(preds_class_gp$upr)),
        col="#ee660050",border=NA)
lines(preds_class_gp$n,preds_class_gp$fit,lty=2)
points(dat_test_class$n,dat_test_class$auc_var,col="green",bg="green",pch=21)

aucs_class_gp<-find_n_threshold(preds_class_gp,
                                round(preds_class_gp$fit,3),
                                thresholds = c(.95,.97,.99))
aucs_class_gp$alg<-"Dayimu et al., 2024"
aucs_class_gp$method<-"gp"

aucs_classification_task<-rbind(aucs_class_nls_fig,aucs_class_nls_day,aucs_class_gp)
aucs_classification_task

# PREDICTION TASK --------------------------------------------------
# Load training points

dbp<-read.csv("data/AFprediction_sample10000_250Hz_50reps.csv")

dbp = data.frame(t(dbp))
auc_varp =rowMeans(dbp[1:50])
np = seq(1000,8000,500)
sep<-apply(dbp[1:50],1,sd)/sqrt(50)
dbp$n = np
dbp$auc_var = auc_varp
dbp$se = sep
dbp_v2=select(dbp, c(n,auc_var,se))
dat_pred<-data.table(dbp_v2)

# Load test points
test_pred<-read.csv("data/AFprediction_sample224000_250Hz_5reps.csv")
dbp_test<-t(test_pred)
auc_varp_test =rowMeans(dbp_test)
np_test<-c(10000,20000,50000,100000,200000)
sep_test<-apply(dbp_test,1,sd)/sqrt(5)
dat_test_pred<-data.frame(n=np_test,auc_var=auc_varp_test,se=sep_test)
rownames(dat_test_pred)<-NULL

# Using Figueroa, 2012 functions ------------------------------------------
start<-c(a=0.01,b=1,c=-0.5) # starting values that can be used also with Dayimu et al. functions  
offset<-0 
i3<-15

# Predict beyond the training points and check against test points
dat_pred_v2<-merge(dat_pred,dat_test_pred[,c("n","auc_var","se")],by=c("n","auc_var","se"),all=TRUE)
dat_pred_v2$w<-ifelse(row(dat_pred_v2[,1])<=i3,1:i3/i3,NA)
res_nls_pred_fig<-FitModel(offset=offset,dat_pred_v2$n,dat_pred_v2$auc_var,
                       dat_pred_v2$w,i = i3,N=nrow(dat_pred_v2),start=start)#false convergence
res_nls_pred_fig
# Use parameters estimated by model with false convergence as starting values 
res_nls_pred_fig_v2<-FitModel(offset=offset,dat_pred_v2$n,dat_pred_v2$auc_var,
                           dat_pred_v2$w,i = i3,N=nrow(dat_pred_v2),start=res_nls_pred_fig$parms)
res_nls_pred_fig_v2

dat_plot_pred<-merge(dat_pred,res_nls_pred_fig_v2$prediction_db,by.x="n",by.y="n",all=TRUE)
plot(dat_plot_pred$n,dat_plot_pred$auc_var,ylim=c(0.5,1))
points(dat_plot_pred$n,dat_plot_pred$predicted,col="red",bg="red",pch=21)
lines(dat_plot_pred$n,dat_plot_pred$predicted.lb)
lines(dat_plot_pred$n,dat_plot_pred$predicted.ub)
polygon(c(dat_plot_pred$n,rev(dat_plot_pred$n)),
        c(dat_plot_pred$predicted.lb,rev(dat_plot_pred$predicted.ub)),
        col="#ee660050",border=NA)
points(dat_test_pred$n,dat_test_pred$auc,pch=21,bg="green")
text(x=c(150000),y=0.6,labels=paste0("RMSE: ",round(res_nls_pred_fig_v2$RMSE,3)))

# Compute n where pre-specified AUCs are reached
extra_data_pred<-data.frame(n=seq(9000,200000,1000),auc_var=NA)
dat_pred_extra<-merge(dat_pred,extra_data_pred,all=TRUE)
dat_pred_extra$w<-ifelse(row(dat_pred_extra[,1])<=i3,1:i3/i3,NA)

res_nls_pred_extra<-FitModel(offset=offset,dat_pred_extra$n,dat_pred_extra$auc_var,
                              dat_pred_extra$w,i = i3,N=nrow(dat_pred_extra),start=res_nls_pred_fig$parms) 

aucs_pred_nls_fig<-find_n_threshold(res_nls_pred_extra$prediction_db,
                                    round(res_nls_pred_extra$prediction_db$predicted,3),
                                    thresholds = c(.80,.82,.83))
aucs_pred_nls_fig$method<-"nls"
aucs_pred_nls_fig$alg<-"Figueroa et al., 2012"

# Using Dayimu, 2024 functions --------------------------------------------
dat_pred_v3<-dat_pred
dat_pred_v3$w<-ifelse(row(dat_pred_v3[,1])<=nrow(dat_pred_v3),1:nrow(dat_pred_v3)/nrow(dat_pred_v3),NA)

# Nonlinear least squares
res_nls_pred_day<-nls_model(dat = dat_pred_v3,auc_var = "auc_var",
                           start=list(alpha = -4.5951199, beta = 1, gamma = 0))#Same values of Figueroa model 
res_nls_pred_day
# Parameters are similar to those of Figueroa:
inv.logit(-1.7821)
inv.logit(-0.1044)

preds_pred_nls<-predict_nls(fit=res_nls_pred_day,dat=dat_pred_v3,auc_var = "auc_var",
                         new_data=data.frame(n = seq(8000, 200000, by = 1000)))%>%
  relocate(n,.before=fit)
preds_pred_nls

dat_comp_nls_pred<-preds_pred_nls[preds_pred_nls$n==10000|preds_pred_nls$n==20000|
                                    preds_pred_nls$n==50000|preds_pred_nls$n==100000|
                                  preds_pred_nls$n==200000,]

rmse(dat_test_pred$auc,dat_comp_nls_pred$fit)

aucs_pred_nls_day<-find_n_threshold(preds_pred_nls,
                                    round(preds_pred_nls$fit,3),
                                    thresholds = c(.80,.82,.83))
aucs_pred_nls_day$method<-"nls"
aucs_pred_nls_day$alg<-"Dayimu et al., 2024"

# Gaussian process
set.seed(123)
res_gp_pred<-gp_model(dat = dat_pred_v3,auc_var = "auc_var",n.chains=5,
                      n.iter=2000000,n.burnin=300000,n.thin=10)
res_gp_pred

preds_pred_gp<-predict_gp(fit=res_gp_pred,dat=dat_pred_v3,auc_var = "auc_var",
                          x_star <- seq(8000, 200000, by = 1000))

plot(dat_pred_v3$n,dat_pred_v3$auc_var,xlim=c(0,200000),ylim=c(0.5,1))
lines(preds_pred_gp$n,preds_pred_gp$fit,lty=2)
lines(preds_pred_gp$n,preds_pred_gp$lwr)
lines(preds_pred_gp$n,preds_pred_gp$upr)
polygon(c(preds_pred_gp$n,rev(preds_pred_gp$n)),
        c(preds_pred_gp$lwr,rev(preds_pred_gp$upr)),
        col="#ee660050",border=NA)
points(dat_test_pred$n,dat_test_pred$auc_var,col="green",bg="green",pch=21)

dat_comp_gp_pred<-preds_pred_gp[preds_pred_gp$n==10000|preds_pred_gp$n==20000|
                                  preds_pred_gp$n==50000|preds_pred_gp$n==100000|
                                  preds_pred_gp$n==200000,]
rmse(dat_test_pred$auc_var,dat_comp_gp_pred$fit)



aucs_pred_gp<-find_n_threshold(preds_pred_gp,
                               round(preds_pred_gp$fit,3),
                               thresholds = c(.80,.82,.83))
  
aucs_pred_gp$alg<-"Dayimu et al., 2024"
aucs_pred_gp$method<-"gp"

aucs_prediction_task<-rbind(aucs_pred_nls_fig,aucs_pred_nls_day,aucs_pred_gp)
aucs_prediction_task


