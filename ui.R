## ui.R ##
library(shinydashboard)


dashboardPage(
  dashboardHeader(),
  dashboardSidebar(
    selectInput(
      "team"
      , "Team:"
      , choices = sort(latestGames$Team)
      , selected = "Utah")
  ),
  dashboardBody(
    fluidRow(
      valueBoxOutput("AvePoints", width = 3)
      ,valueBoxOutput("AvePA", width = 3)
      ,valueBoxOutput("Wins", width = 3)
      ,valueBoxOutput("Losses", width = 3)
    )
    ,fluidRow(
      box(DT::dataTableOutput("table"), width = 12)
      # ,box(plotOutput("graph"))
    )
  )
)
   