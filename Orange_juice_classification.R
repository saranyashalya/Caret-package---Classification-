##https://www.machinelearningplus.com/machine-learning/caret-package/


library(caret) #Machine learning model implementaiton
library(caTools) # train test data split
library(skimr) # Descriptive statistics
library(RANN)

orange = read.csv("C:/Users/H303937/Downloads/orange_juice_withmissing.csv")

str(orange)

head(orange)

summary(orange)

#number of missing elements 
sapply(orange, function(x) sum(is.na(x)))


##spliting into train and test dataset
set.seed(100)
sample = sample.split(orange$Purchase, SplitRatio = 0.8)
train = orange[sample==TRUE,]
test = orange[sample==FALSE,]

#separating X and Y for later use
X = train[,2:18]
Y = train[1]


## Viewing descriptive statistics
skimmed = skim_to_wide(train)
skimmed[,c(1:5,9:11, 13, 15:16)]

skimmed_list = skim_to_list(train)
skimmed_list

# Preprocessing using caret
## using knn imputation for misisng values

preprocessed_data <- preProcess(train, method=c("knnImpute"))
preprocessed_data


sapply(train, function(x) sum(is.na(x)))

#predict the missing value for train
train = predict(preprocessed_data, newdata = train)
anyNA(train)

## One hot encoding
dummies_model <- dummyVars(Purchase ~ ., train)
train <- predict(dummies_model, train)

#converting into dataframe
train <- as.data.frame(train)

str(train) #Note purchase columns is not available.


## Transforming data
preprocess_range <- preProcess(train, method="range")
train <- predict(preprocess_range, newdata = train)

##Adding Purchase column
train$Purchase <- Y$Purchase

#checking min and max value column wise
apply(train, 2, function(x) c("min"=min(x), "max"= max(x)))


##featureplot

featurePlot(x = train[,0:18], y= train$Purchase, plot = "box")

featurePlot(x = train[,0:18], y=train$Purchase, plot='density')


## RFE - Recursive Feature Elimination

set.seed(100)
options(warn=-1)

subsets = c(1:5, 10,15,18)
ctrl <- rfeControl(functions = rfFuncs,
                   method ="repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=train[, 1:18], y=train['Purchase'],
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

##training and tuning model

#model list available in caret

modelNames <- paste(names(getModelInfo()), collapse=',')
modelNames

modelLookup('earth')

##Applying MARS - Multivariate Adaptive Regression Splines

set.seed(100)

#train the modle
model_mars <- train(Purchase ~., data = train, metho='earth')

model_mars

plot(model_mars,main ="Model Accuracy with MARS")


## variable Importance using mars

varimp_mars <- varImp(model_mars)

plot(varimp_mars, main ="Variable importance using mars")

##Preprocessing test dataset


test1 <- predict(preprocessed_data, newdata = test)

test2 <- predict(dummies_model, newdata=test1)

test3 <- predict(preprocess_range, newdata = test2)

apply(test3, 2,  function(x) c('min'=min(x), 'max'=max(x)) )

head(test3)


##Predict on testdata

preds <- predict(model_mars, test3)

##Confusion matrix

confusionMatrix(test$Purchase, preds, positive = "MM")


##Hyperparameter tuning

set.seed(100)
trainCtrl = trainControl(method = 'cv', number = 5, savePredictions = 'final',
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary) 

model_mars2 <- train(Purchase ~ ., data = train, method = 'earth', trControl = trainCtrl, tuneLength = 5,
                     metric ='ROC')

model_mars2

##Predicting on test dataset
preds = predict(model_mars2, newdata = test3)

confusionMatrix(test$Purchase, preds, positive = "MM")


## Hyper parameter tuning using tuneGrid

marsgrid <- expand.grid('nprune'= c(2,4,6,8,10), degree = c(1,2,3))

set.seed(100)
model_mars3 <- train(Purchase ~., data=train, method ='earth', tuneGrid = marsgrid,
                     metric = 'ROC', trControl = trainCtrl)

model_mars3

##Predicting on test set

preds = predict(model_mars3, newdata = test3)
confusionMatrix(test$Purchase, preds, positive = "MM")


## Training more models:

##ada boost
set.seed(100)
model_adaboost <- train(Purchase ~., data= train, method ='adaboost', tuneLength =2, 
                        trControl = trainCtrl)

model_adaboost

## random forest

set.seed(100)
model_rf <- train(Purchase ~., data = train, method = 'rf', tuneLength =2, 
                  trControl = trainCtrl)

model_rf


## xgboost DART

# set.seed(100)
# model_xgbDART <- train(Purchase ~., data = train, method='xgbDART', tuneLength=5, 
#                        trControl = trainCtrl)
# 
# model_xgbDART

## SVM

set.seed(100)

model_svmRadial <- train(Purchase ~., data = train, method='svmRadial', tuneLength= 15,
                         trControl = trainCtrl)

model_svmRadial

## Compare model performances

models_compare <- resamples(list(ADABOOST = model_adaboost, RF = model_rf, 
                                  MARS = model_mars3,
                                 SVM = model_svmRadial))

summary(models_compare)

## Boxplots to compare model

scales <- list(x = list(relation ="free"), y = list(relation ="free"))

bwplot(models_compare, scales)

### Ensembling the predictions

library(caretEnsemble)


trainCtrl <- trainControl(method='repeatedcv', number = 10,repeats = 5 ,
                          savePredictions = TRUE, classProbs = TRUE)


algorithm_list <-  c('rf','adaboost','svmRadial', 'earth')

set.seed(100)

models <- caretList(Purchase ~., data = train, methodList = algorithm_list, trControl = trainCtrl)

results <- resamples(models)

summary(results)


##boxplot of compare models

scales <- list(x = list(relation ='free'), y =list(relation ='free'))

bwplot(results, scales = scales)
