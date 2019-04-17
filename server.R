library(tidyverse)
library(shiny)
library(xgboost)
server <- function(input, output) {
  newPredict <- reactive({
    data.frame(
      Sepal.Length=input$sep.length, 
      Sepal.Width=input$sep.width,
      Petal.Length=input$p.length, 
      Petal.Width=input$p.width)
  
  })
  
  output$predictions <- DT::renderDataTable({
    data.frame(
      Iris = c("setosa", "versicolor", "virginica")
      ,Probability = sort(predict(IrisModel, as.matrix(newPredict())), decreasing=TRUE)
    )
    
  })
  output$scatter<-renderPlot({
    plot(x=iris$Sepal.Length, y=iris$Sepal.Width, 
         xlab="Sepal Length", ylab="Sepal Width",  main="Sepal Length-Width")
    points(newPredict(), col="red", pch=19)
  })
  output$den1<-renderPlot({
    d <- density(iris$Sepal.Length)
    hist(iris$Sepal.Length, breaks=12, prob=TRUE, xlab="Sepal Length", main="Histogram & Density Curve")
    lines(d, lty=2, col="green")
    abline(v=newPredict()$Sepal.Length, col='red')
  })
  output$den2<-renderPlot({
    d <- density(iris$Sepal.Width)
    hist(iris$Sepal.Width, breaks=12, prob=TRUE, xlab="Sepal Width", main="Histogram & Density Curve")
    lines(d, lty=2, col="green")
    abline(v=newPredict()$Sepal.Width, col='red')
  })
  output$den3<-renderPlot({
    d <- density(iris$Petal.Length)
    hist(iris$Petal.Length, breaks=12, prob=TRUE, xlab="Petal Length", main="Histogram & Density Curve")
    lines(d, lty=2, col="green")
    abline(v=newPredict()$Petal.Length, col='red')
  })
  output$den4<-renderPlot({
    d <- density(iris$Petal.Width)
    hist(iris$Petal.Width, breaks=12, prob=TRUE, xlab="Petal Width", main="Histogram & Density Curve")
    lines(d, lty=2, col="green")
    abline(v=newPredict()$Petal.Width, col='red')
  })
}