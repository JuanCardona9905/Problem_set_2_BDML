#main script
#este código contiene todos los scripts utilizados para el problem set 1

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
       readxl, # importar Excel
       writexl, # exportar Excel
       boot, # bootstrapping
       ggpubr, # extensiones de ggplot2
       WVPlots, # gráficos de variables ponderadas
       patchwork, # para combinar gráficos
       gridExtra, # para combinar gráficos
       ggplot2, # gráficos
       caret, # para evaluación de modelos predictivos
       data.table) # para manipulación de datos

#se define la ruta de trabajo
ifelse(grepl("camilabeltran", getwd()),
       wd <- "/Users/camilabeltran/OneDrive/Educación/PEG - Uniandes/BDML/GitHub/problem_set/Problem_set_1",
       ifelse(grepl("Juan",getwd()),
              wd <- "C:/Users/Juan/Documents/Problem_set_1",
              ifelse(grepl("juanp.rodriguez",getwd()),
                     wd <- "C:/Users/juanp.rodriguez/Documents/GitHub/Problem_set_1",
                     ifelse(grepl("C:/Users/User",getwd()),
                            wd <- "C:/Users/User/OneDrive - Universidad de los andes/Big Data y Machine Learning/Problem_set_1/Problem_set_1",
                            ifelse(grepl("/Users/aleja/",getwd()),
                                   wd <- "/Users/aleja/Documents/Maestría Uniandes/Clases/Big Data y Machine Learning/Repositorios Git Hub/Problem_set_1)",
                                   wd <- "otro_directorio")))))

#Script: "01_web_scraping.R". Realiza el proceso de web scraping para conseguir los datos
setwd(paste0(wd,"/scripts"))
source("01_web_scraping.R")
#Script: "02_Data.R". Realiza la limpieza de la base de datoć y estadísticas descriptivas
setwd(paste0(wd,"/scripts"))
source("02_data.R")
#Script: "03_Data.R". Realiza estimaciones del perfil edad-ingreso
setwd(paste0(wd,"/scripts"))
source("03_age_wage_profile.R")
#Script: "04_The gender earnings GAP.R". Realiza estimaciones de la brecha salarial por género
setwd(paste0(wd,"/scripts"))
source("04_gender_earnings_gap.R")
#Script: "05_Predicting_Earnings.R". Realiza predicciones y estimaciones de cross-validation
setwd(paste0(wd,"/scripts"))
source("05_predicting_earnings.R")
