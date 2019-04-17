library(xgboost)
library(tidyverse)
library(DiagrammeR)
library(data.table)


y=as.numeric(iris$Species)-1

x<-select(iris, -Species)

var.names=names(x)

x<-as.matrix(x)

#Making a list

param <- list(
  "objective" = "multi:softprob"
  ,"eval_metric" = "mlogloss"
  ,"num_class" = length(table(y))
  ,"eta" = .05
  ,"max_depth" = 7
  # ,"lambda" = 1
  #,"alpha" = .8
  #,"min_child_weight" = 3
  #,"subsample" = .9
  # ,"colsample_bytree" = .6
)

#150 rounds 
cv.nround = 250


bst.cv <- xgb.cv(param = param, data = x, label = y
                 ,nfold = 3, nrounds = cv.nround
                 ,missing = NA, prediction = TRUE)

nround = which(bst.cv$evaluation_log$test_mlogloss_mean == min(bst.cv$evaluation_log$test_mlogloss_mean))


##  check overfitting
ggplot(bst.cv$evaluation_log, aes(x = iter)) +
  geom_line(aes(y = train_mlogloss_mean), color = "blue") +
  geom_line(aes(y = test_mlogloss_mean), color = 'red') +
  geom_vline(xintercept = nround, linetype = 3)


## Building the main classifier
IrisClassifier <- xgboost(params = param, data = x, label = y
                          ,nrounds = nround, missing = NA)


xgb.importance(feature_names = var.names, model = IrisClassifier)


xgboost::xgb.plot.tree(IrisClassifier, feature_names = var.names, n_first_tree = 1)

## Generating predictions
p<-predict(IrisClassifier,x)
matrix(p,ncol=length(table(y)),byrow=TRUE)


xgb.save(model=IrisClassifier,fname='iris.model')