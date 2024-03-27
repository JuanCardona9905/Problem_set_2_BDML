##################################################
###################### Data ######################
##################################################

#### 1. Cargar Paquetes ----
{
#se borra la memoria
rm(list = ls())
#se cargan los paquetes
library(pacman)
p_load(rio, # importación/exportación de datos
       tidyverse, # datos ordenados (ggplot y Tidyverse)
       skimr, # datos de resumen
       visdat, # visualización de datos faltantes
       corrplot, # gráficos de correlación
       stargazer, # tablas/salida a TEX.
       rvest, # web-scraping
       readxl,
       readr, # importar Excel
       writexl, # exportar Excel
       boot, # bootstrapping
       ggpubr, # extensiones de ggplot2
       WVPlots, # gráficos de variables ponderadas
       patchwork, # para combinar gráficos
       gridExtra, # para combinar gráficos
       ggplot2, # gráficos
       caret, # para evaluación de modelos predictivos
       glmnet, # para evaluación de modelos predictivos
       data.table, # para manipulación de datos
       MASS, # El paquete tiene a la funcion de LDA
       class, # El paquete tiene a la funcion de k-neighbours
       naniar) # missing

#se define la ruta de trabajo
ifelse(grepl("camilabeltran", getwd()),
       wd <- "/Users/camilabeltran/OneDrive/Educación/PEG - Uniandes/BDML/GitHub/problem_set/Problem_set_1",
       ifelse(grepl("Juan",getwd()),
              wd <- "C:/Users/Juan/Documents/Problem_set_2",
              ifelse(grepl("juanp.rodriguez",getwd()),
                     wd <- "C:/Users/juanp.rodriguez/Documents/GitHub/Problem_set_1",
                     ifelse(grepl("C:/Users/User",getwd()),
                            wd <- "C:/Users/User/OneDrive - Universidad de los andes/Big Data y Machine Learning/Problem_set_1/Problem_set_1",
                            ifelse(grepl("/Users/aleja/",getwd()),
                                   wd <- "/Users/aleja/Documents/Maestría Uniandes/Clases/Big Data y Machine Learning/Repositorios Git Hub/Problem_set_1)",
                                   wd <- "otro_directorio")))))

#Script: "01_web_scraping.R". Realiza el proceso de web scraping para conseguir los datos
setwd(paste0(wd,"/scripts"))
}

#### 2. Importar bases de datos ----
{
### Importar las bases de datos finales 
# Train

# Test
  
}


#### 3. Modelos ----

### Logistic Regression --

### Modelo 1 -
{
## Entrenamiento

# Modelo
Mod_1_LR <- glm(formula = Pobre ~., 
                data = train, 
                family = binomial)
summary(Mod_1_LR)

# Probabilidades in-sample
Mod_1_LR_prob <- predict(Mod_1_LR, type="response")
# Hacer un histograma de las probabilidades

# Prediccion in-sample
Mod_1_LR_pred <- ifelse(Mod_1_LR_prob>0.5, "Pobre", "No Pobre")

# Matriz de confusion
table(Mod_1_LR_pred, train$Pobre) #aca se puede ajustar el umbral
mean(Mod_1_LR_pred == train$Pobre)

# Probabilidades out-sample
Mod_1_LR_prob <- predict(Mod_1_LR, newdata = test, type="response")
# Hacer un histograma de las probabilidades

# Prediccion out-sample
Mod_1_LR_pred <- ifelse(Mod_1_LR_prob>0.5, "Pobre", "No Pobre")

}

### Linear Discriminant Analysis --

### Modelo 2 -
{
  ## Entrenamiento
  
  # Modelo
  Mod_2_LDA <- lda(formula = Pobre ~., 
                  data = train)
  Mod_2_LDA
  plot(Mod_2_LDA)

  # Prediccion in-sample
  Mod_2_LDA_pred <- predict(Mod_2_LDA, train)
  
  # Matriz de confusion
  table(Mod_2_LDA_pred, train$Pobre) #aca se puede ajustar el umbral
  mean(Mod_1_LR_pred == train$Pobre)
  
  # Prediccion out-sample
  Mod_2_LDA_pred <- predict(Mod_2_LDA, test)
  # Borrar Creo que toca con data.frame...
}

### K-nearest Neighbour Classification --

### Modelo 3 -
{
# Es necesario los datos en forma de matriz 
X_train <- as.matrix(cbind("aqui van las covariables"))
X_test <- as.matrix(cbind("aqui van las covariables"))

# Prediccion 
Mod_3_KNN_pred <- knn(X_train, X_test, train$Pobre, k=1)

table(Mod_3_KNN_pred, train$Pobre)
}

### Elastic Net --

### Modelo 4 -
{
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      classProbs = TRUE,
                      savePredictions = T)
  # Modelo
  set.seed(098063)
  Mod_4_EN <- train(Pobre~.,
                  data=train,
                  metric = "Accuracy",
                  method = "glmnet",
                  trControl = ctrl,
                  tuneGrid=expand.grid(
                    alpha = seq(0,1,by=.2),
                    lambda =10^seq(10, -2, length = 10)
                  ))
  # Resultados 
  Mod_4_EN
  # Se escogieron los paramentros ...
}
