library(caret)
library(ggplot2)
library(glmnet)

library(bootStepAIC)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)
library(data.table)
library(mltools)
library(imputeTS)
library(forcats)
library(fastDummies)

setwd("C:/Users/neela/Desktop/My Stuff/Queens MMA/MMA 867- Predictive Modeling/Assignments/Assignment 1/House Prices")
train<-fread("train.csv", header=TRUE, sep=",") [,-1]
validation<-fread("test.csv", header=TRUE, sep=",")[,-1] 

head(train)
str(train)
tail(train)
summary(train)

train_numeric<-train%>%
  select_if(is.integer)

train_numeric<-na_mean(train_numeric)

train_factor<-train%>%
  select_if(is.character)

train_factor <- data.frame(lapply(train_factor, function(x) as.factor(as.character(x))))

train_factor<-dummy_columns(train_factor)

train_factor<-train_factor%>%
  select(44:311)

hist(train$SalePrice,xlab = "Sale Price",main = "Sale Price Distribution")

train_numeric <- log(train_numeric+1)

hist(train_numeric$SalePrice,xlab = "Log Sale Price",main = "Log Sale Price Distribution")

train<-cbind(train_numeric,train_factor)
train[is.na(train)] <- 0

##############################################################################################################

set.seed(123)
sample<-sample.int(n=nrow(train),size = floor(0.7*nrow(train)),replace=F)
train_set<-train[sample,]
test_set<-train[-sample,]

model1<-lm(SalePrice~.,data=train_set)
summary(model1)

#model1_Improved<-stepAIC(model1,direction='both')
#summary(model1_Improved)

#Model from stepAIC 
model1_Improved<-lm(formula = SalePrice ~ LotArea + OverallQual + OverallCond + 
                      YearBuilt + YearRemodAdd + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + 
                      TotalSF + LowQualFinSF + GrLivArea + BsmtFullBath + HalfBath + 
                      Fireplaces + GarageCars + WoodDeckSF + `3SsnPorch` + ScreenPorch + 
                      PoolArea + `MSZoning_C (all)` + MSZoning_FV + Alley_Pave + 
                      LandContour_Bnk + LandContour_Low + Utilities_AllPub + LotConfig_Corner + 
                      LotConfig_CulDSac + LotConfig_FR2 + LandSlope_Gtl + LandSlope_Mod + 
                      Neighborhood_Blmngtn + Neighborhood_CollgCr + Neighborhood_Crawfor + 
                      Neighborhood_Edwards + Neighborhood_Gilbert + Neighborhood_IDOTRR + 
                      Neighborhood_MeadowV + Neighborhood_Mitchel + Neighborhood_NAmes + 
                      Neighborhood_NoRidge + Neighborhood_NPkVill + Neighborhood_NridgHt + 
                      Neighborhood_NWAmes + Neighborhood_OldTown + Neighborhood_Sawyer + 
                      Neighborhood_StoneBr + Condition1_Artery + Condition1_Feedr + 
                      Condition1_PosA + Condition1_RRAe + Condition1_RRNe + Condition2_Feedr + 
                      Condition2_PosN + BldgType_1Fam + HouseStyle_2.5Unf + HouseStyle_2Story + 
                      RoofStyle_Flat + RoofStyle_Gable + RoofMatl_ClyTile + RoofMatl_CompShg + 
                      Exterior1st_AsbShng + Exterior1st_BrkComm + Exterior1st_BrkFace + 
                      Exterior1st_MetalSd + Exterior2nd_AsbShng + Exterior2nd_BrkFace + 
                      Exterior2nd_CmentBd + Exterior2nd_Stone + Exterior2nd_VinylSd + 
                      ExterQual_Ex + ExterCond_Ex + Foundation_BrkTil + Foundation_CBlock + 
                      Foundation_PConc + Foundation_Slab + Foundation_Stone + BsmtQual_Fa + 
                      BsmtQual_Gd + BsmtQual_TA + BsmtCond_Po + BsmtExposure_Av + 
                      BsmtExposure_Gd + BsmtExposure_Mn + BsmtFinType1_ALQ + BsmtFinType1_BLQ + 
                      BsmtFinType1_LwQ + BsmtFinType1_Rec + BsmtFinType2_ALQ + 
                      BsmtFinType2_GLQ + BsmtFinType2_LwQ + BsmtFinType2_Rec + 
                      Heating_GasA + Heating_Grav + Heating_OthW + HeatingQC_Ex + 
                      HeatingQC_Fa + CentralAir_N + KitchenQual_Ex + Functional_Maj2 + 
                      Functional_Min1 + Functional_Min2 + Functional_Mod + Functional_Sev + 
                      FireplaceQu_Gd + FireplaceQu_TA + GarageType_CarPort + GarageQual_Ex + 
                      GarageQual_Po + GarageCond_Ex + GarageCond_Fa + GarageCond_Po + 
                      PoolQC_Fa + Fence_GdWo + Fence_MnWw + SaleType_Con + SaleType_ConLD + 
                      SaleType_CWD + SaleType_New + SaleCondition_AdjLand + SaleCondition_Normal + 
                      GarageFinish_Unf + Exterior2nd_MetalSd + MasVnrType_NA + 
                      BsmtCond_TA, data = train_set)



test_set_model1<-test_set
test_set_model1$Prediction<-exp(predict(model1,test_set_model1))
test_set_model1$SalePrice<-exp(test_set_model1$SalePrice)

test_set_model1$percent_errors <- abs((test_set_model1$SalePrice-test_set_model1$Prediction)/test_set_model1$SalePrice)*100
mean(test_set_model1$percent_errors) #display Mean Absolute Percentage Error (MAPE)
rmsle(test_set_model1$SalePrice,test_set_model1$Prediction)


#Model for Validation Set 
model1_Improved_validation<-lm(formula = SalePrice ~ LotArea + OverallQual + OverallCond + 
                                 YearBuilt + YearRemodAdd + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + 
                                 TotalSF + LowQualFinSF + GrLivArea + BsmtFullBath + HalfBath + 
                                 Fireplaces + GarageCars + WoodDeckSF + `3SsnPorch` + ScreenPorch + 
                                 PoolArea + MSZoning_FV + Alley_Pave + 
                                 LandContour_Bnk + LandContour_Low + Utilities_AllPub + LotConfig_Corner + 
                                 LotConfig_CulDSac + LotConfig_FR2 + LandSlope_Gtl + LandSlope_Mod + 
                                 Neighborhood_Blmngtn + Neighborhood_CollgCr + Neighborhood_Crawfor + 
                                 Neighborhood_Edwards + Neighborhood_Gilbert + Neighborhood_IDOTRR + 
                                 Neighborhood_MeadowV + Neighborhood_Mitchel + Neighborhood_NAmes + 
                                 Neighborhood_NoRidge + Neighborhood_NPkVill + Neighborhood_NridgHt + 
                                 Neighborhood_NWAmes + Neighborhood_OldTown + Neighborhood_Sawyer + 
                                 Neighborhood_StoneBr + Condition1_Artery + Condition1_Feedr + 
                                 Condition1_PosA + Condition1_RRAe + Condition1_RRNe + Condition2_Feedr + 
                                 Condition2_PosN + BldgType_1Fam + HouseStyle_2.5Unf + HouseStyle_2Story + 
                                 RoofStyle_Flat + RoofStyle_Gable + RoofMatl_CompShg + 
                                 Exterior1st_AsbShng + Exterior1st_BrkComm + Exterior1st_BrkFace + 
                                 Exterior1st_MetalSd + Exterior2nd_AsbShng + Exterior2nd_BrkFace + 
                                 Exterior2nd_CmentBd + Exterior2nd_Stone + Exterior2nd_VinylSd + 
                                 ExterQual_Ex + ExterCond_Ex + Foundation_BrkTil + Foundation_CBlock + 
                                 Foundation_PConc + Foundation_Slab + Foundation_Stone + BsmtQual_Fa + 
                                 BsmtQual_Gd + BsmtQual_TA + BsmtCond_Po + BsmtExposure_Av + 
                                 BsmtExposure_Gd + BsmtExposure_Mn + BsmtFinType1_ALQ + BsmtFinType1_BLQ + 
                                 BsmtFinType1_LwQ + BsmtFinType1_Rec + BsmtFinType2_ALQ + 
                                 BsmtFinType2_GLQ + BsmtFinType2_LwQ + BsmtFinType2_Rec + 
                                 Heating_GasA + Heating_Grav + HeatingQC_Ex + 
                                 HeatingQC_Fa + CentralAir_N + KitchenQual_Ex + Functional_Maj2 + 
                                 Functional_Min1 + Functional_Min2 + Functional_Mod + Functional_Sev + 
                                 FireplaceQu_Gd + FireplaceQu_TA + GarageType_CarPort + 
                                 GarageQual_Po + GarageCond_Ex + GarageCond_Fa + GarageCond_Po + 
                                 Fence_GdWo + Fence_MnWw + SaleType_Con + SaleType_ConLD + 
                                 SaleType_CWD + SaleType_New + SaleCondition_AdjLand + SaleCondition_Normal + 
                                 GarageFinish_Unf + Exterior2nd_MetalSd + MasVnrType_NA + 
                                 BsmtCond_TA, data = train)


validation_numeric<-validation%>%
  select_if(is.integer)

validation_numeric<-na_mean(validation_numeric)

validation_factor<-validation%>%
  select_if(is.character)

validation_factor <- data.frame(lapply(validation_factor, function(x) as.factor(as.character(x))))

validation_factor<-dummy_columns(validation_factor)

validation_factor<-validation_factor%>%
  select(44:299)

validation_numeric <- log(validation_numeric+1)

validation<-cbind(validation_numeric,validation_factor)

validation[is.na(validation)] <- 0

validation$SalePrice<-exp(predict(model1_Improved_validation,validation))

validation<-validation%>%
  select(294)

validation_clean<-fread("test.csv", header=TRUE, sep=",")

validation_clean<-validation_clean%>%
  select(1)

validation_final<-cbind(validation_clean,validation)

setwd("C:/Users/neela/Desktop")

write.csv(validation_final,"validation_model2.csv",row.names = F)

##############################################################################################################

set.seed(123)
sample<-sample.int(n=nrow(train_numeric),size = floor(0.7*nrow(train_numeric)),replace=F)
train_set<-train_numeric[sample,]
test_set<-train_numeric[-sample,]

model2<-lm(SalePrice~., data=train_set)
summary(model2)

model2_Improved<-stepAIC(model2,direction='both')
summary(model2_Improved)

test_set_model2<-test_set
test_set_model2$Prediction<-exp(predict(model2_Improved,test_set_model2))
test_set_model2$SalePrice<-exp(test_set_model2$SalePrice)

test_set_model2$percent_errors <- abs((test_set_model2$SalePrice-test_set_model2$Prediction)/test_set_model2$SalePrice)*100
mean(test_set_model2$percent_errors) #display Mean Absolute Percentage Error (MAPE)
rmsle(test_set_model2$SalePrice,test_set_model2$Prediction)

model2<-lm(SalePrice~., data=train_numeric)
summary(model2)

model2_Improved<-stepAIC(model2,direction='both')
summary(model2_Improved)

validation_numeric<-validation%>%
  select_if(is.integer)

validation_numeric<-na_mean(validation_numeric)

validation_numeric <- log(validation_numeric+1)

validation_numeric$SalePrice<-exp(predict(model2_Improved,validation_numeric))

validation_numeric<-validation_numeric%>%
  select(38)

validation_clean<-fread("test.csv", header=TRUE, sep=",")

validation_clean<-validation_clean%>%
  select(1)

validation_final<-cbind(validation_clean,validation_numeric)

setwd("C:/Users/neela/Desktop")

write.csv(validation_final,"validation_model1.csv",row.names = F)

##############################################################################################################

#create the y variable and matrix (capital X) of x variables (will make the code below easier to read + will ensure that all interactoins exist)
y<-train_set$SalePrice
X<-model.matrix(Id~LotArea + OverallQual + OverallCond + 
                  YearBuilt + YearRemodAdd + BsmtFinSF1 + BsmtFinSF2 + X2ndFlrSF + 
                  TotalSF + LowQualFinSF + GrLivArea + BsmtFullBath + FullBath + 
                  HalfBath + BedroomAbvGr + KitchenAbvGr + TotRmsAbvGrd + Fireplaces + 
                  GarageCars + GarageArea + WoodDeckSF + ScreenPorch + PoolArea + 
                  YrSold,train_numeric)

X<-cbind(train_numeric$Id,X)

# split X into testing, training/holdout and prediction as before
sample<-sample.int(n=nrow(train_numeric),size = floor(0.7*nrow(train_numeric)),replace=F)
X.training<-X[sample,]
X.testing<-X[-sample,]

#LASSO (alpha=1)
lasso.fit<-glmnet(x = X.training, y = y, alpha = 1)
plot(lasso.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.training, y = y, alpha = 1) #create cross-validation data
plot(crossval)
penalty.lasso <- crossval$lambda.min #determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph
plot(crossval,xlim=c(-8.5,-6),ylim=c(0.006,0.008)) # lets zoom-in
lasso.opt.fit <-glmnet(x = X.training, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lasso.opt.fit) #resultant model coefficients

# predicting the performance on the testing set
lasso.testing <- predict(lasso.opt.fit, s = penalty.lasso, newx =X.testing)
mean(abs(lasso.testing-test_set$SalePrice/test_set$SalePrice)*100) #calculate and display MAPE
rmsle(test_set$SalePrice,lasso.testing)
