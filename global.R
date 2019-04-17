library(DT)
library(ggplot2)
library(shiny)
library(xgboost)
library(tidyverse)
library(DiagrammeR)
library(data.table)


IrisModel<-xgb.load("iris.model")