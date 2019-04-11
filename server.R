library(tidyverse)

server <- function(input, output) { 
  
  DF <- reactive({
    filter(latestGames, Team == input$team)
  })
  
  output$table <- DT::renderDataTable({
    DF()
  }#, options = list(autoWidth = TRUE)
  )
  
  
  output$AvePoints <- renderValueBox({
    valueBox(
      round(mean(DF()$PF), 2)
      ,"Average Points"
    )
  })
  
  output$AvePA <- renderValueBox({
    valueBox(
      round(mean(DF()$PA), 2)
      ,"Average Points Allowed"
    )
  })
  
  output$Wins <- renderValueBox({
    valueBox(
      sum(DF()$WL == 'W')
      ,"Wins"
    )
  })
  
  
  output$Losses <- renderValueBox({
    valueBox(
      sum(DF()$WL == 'L')
      ,"Losses"
    )
  })
  
}


