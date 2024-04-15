#################################################
#### Estadística Descriptiva Ingreso Hogar #####
#################################################

#### 1. Importar bases de datos ----
{
  ### Importar las bases de entrenamiento
  # Personas
  setwd(paste0(wd,"/Data"))
  load(file = "train_personas.RData")
  length(train_personas) # 135 variables 
  nrow(train_personas) # 543.109 observaciones
  
  # Hogares
  train_hogares <- read_csv("train_hogares.csv")
  length(train_hogares) # 23 variables 
  nrow(train_hogares) # 164.960 observaciones
  
  ### Importar las bases de testeo 
  # Personas
  test_personas <- read_csv("test_personas.csv")
  length(test_personas) # 63 variables 
  nrow(test_personas) # 219.644 observaciones
  
  
  
  # Hogares
  test_hogares <- read_csv("test_hogares.csv")
  length(test_hogares) # 16 variables 
  nrow(test_hogares) # 66.168 observaciones
  
  ### Variables importantes 
  colnames(train_personas)
  colnames(test_personas)
  # id es el hogar 
  # El orden es la persona especifica del hogar
  
  colnames(train_hogares)
  colnames(test_hogares)
  # Los hogares no tienen la variable Orden porque las personas estan 
  # agrupadas o colapsadas en id
}

#### 3. Grafico principal (histograma ingreso) ----
{ # Grafica que divide a pobres
  summary(train_hogares$Ingtotug)
  
  train_hogares_Graf <-  train_hogares %>% 
    filter(Ingtotug < 2000000) %>% 
    mutate(y = density(Ingtotug))
  
  #Grafico base
  ggplot(train_hogares_Graf, aes(x = Ingpcug)) +
    geom_density(fill = "cyan4", alpha = 0.5) +
    geom_vline(xintercept = mean(train_hogares$Lp), color = "red", linetype = "dashed") +
    labs(x = "Ingreso", y = "Densidad") +
    theme_minimal()
  
  media_ingreso <- mean(train_hogares$Lp)
  ggplot(train_hogares_Graf, aes(x = (Ingpcug/1000000))) +
    geom_histogram(data = subset(train_hogares_Graf, (Ingpcug/1000000) < (media_ingreso/1000000)), fill = "#B22222", bins = 40,alpha = .6) +
    geom_histogram(data = subset(train_hogares_Graf, (Ingpcug/1000000) >= (media_ingreso/1000000)), fill = "#36648B", bins = 40, alpha = .8) +
    geom_vline(aes(xintercept = (media_ingreso/1000000)), color = "yellow3", linetype = "dashed", size = 1) +
    theme_minimal() +
    labs(title = "Distribución del ingreso del hogar",
         subtitle = "Observaciones por debajo y por encima del umbral de pobreza",
         x = "Ingreso del hogar ($ millones)",
         y = "Frecuencia", plot.title = element_text(hjust = 0.5))
}

#### 3. Pruebas de grafico de densidad ----
{
  table(train_hogares$Pobre)
  prop.table(table(train_hogares$Pobre))
  
  
  library(car)
  
  densityPlot(x = train_hogares_Graf$Ingtotug, g = as.factor(train_hogares_Graf$Pobre), method=c("adaptive", "kernel"),
              bw=if (method == "adaptive") bw.nrd0 else "SJ", adjust=1,
              kernel, xlim, ylim,
              normalize=FALSE, xlab=deparse(substitute(x)), ylab="Density", main="",
              col=carPalette(), lty=seq_along(col), lwd=2, grid=TRUE,
              legend=TRUE, show.bw=FALSE, rug=TRUE)
  
  densityPlot(x = train_hogares_Graf$Ingtotug, g = as.factor(train_hogares_Graf$Pobre),col=carPalette())
  
  
  # Create overlaid density plots for same data
  ggplot(train_hogares_Graf, aes(x = Ingtotug, fill = as.factor(Pobre))) +
    geom_density(alpha = .3)
}

