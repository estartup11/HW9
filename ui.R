library(shinydashboard)


dashboardPage(
  dashboardHeader(),
  dashboardSidebar(
    numericInput(
      "sep.length"
      , "Sepal Length:"
      , 7)
    ,
    numericInput(
      "sep.width"
      ,'Sepal Width:'
      ,3)
    ,
    numericInput(
      "p.length"
      ,'Petal Length:'
      ,4)
    ,
    numericInput(
      "p.width"
      ,'Petal Width:'
      ,.7)
  ),
  dashboardBody(
    fluidRow(
      box(DT::dataTableOutput("predictions"))
      # ,box(plotOutput("graph"))
    ),
    fluidRow(
      tabPanel( 'Plot', plotOutput('scatter'))
    ),
    fluidRow(
      tabPanel("Plot", plotOutput("den1"))
    ),
    fluidRow(
      tabPanel("Plot", plotOutput("den2"))
    ),
    fluidRow(
      tabPanel("Plot", plotOutput("den3"))
    ) ,
    fluidRow(
      tabPanel("Plot", plotOutput("den4"))
    )
  )
)