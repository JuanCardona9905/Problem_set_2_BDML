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
### Importar las bases de entrenamiento 
# Personas
load(file = "train_personas.RData")
View(train_personas)
length(train_personas) # 135 variables 
nrow(train_personas) # 543.109 observaciones



# Hogares
train_hogares <- read_csv("~/Problem_set_2/Data/train_hogares.csv")
View(train_hogares)
length(train_hogares) # 23 variables 
nrow(train_hogares) # 164.964 observaciones

### Importar las bases de testeo 
# Personas
test_personas <- read_csv("~/Problem_set_2/Data/test_personas.csv")
View(test_personas)
length(test_personas) # 63 variables 
nrow(test_personas) # 219.644 observaciones
# Hogares
test_hogares <- read_csv("~/Problem_set_2/Data/test_hogares.csv")
View(test_hogares)
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

#### 3. Modificacion base de datos ----

### Hogares --
{ #Train
train_hogares <- train_hogares %>% 
  mutate(
    Cabecera = ifelse(Clase==1,1,0), 
    DormitorXpersona = P5010/Nper,
    Ln_Cuota = log(P5100), #Log de pago de cuota
    Ln_Est_arrien = log(P5130), #Log de pago de estimativo de pago de arriendo
    Ln_Pago_arrien = log(P5140)) %>% #Log de pago arriendo
   
    rename(Ing_tot_hog  = Ingtotugarr,
           N_cuartos_hog =  P5000,
           Ocup_vivienda = P5090) %>% 
    
    select(-Ingtotug, # Ingreso antes de arriendo
           -Ingpcug,
           -Li,
           -Indigente,
           -Nindigentes,
           -Npobres,
           -P5010,
           -P5100,
           -P5130,
           -P5140) 
colnames(train_hogares)
}

{ # Test
  test_hogares <- test_hogares %>% 
    mutate(
      Cabecera = ifelse(Clase==1,1,0), 
      DormitorXpersona = P5010/Nper,
      Ln_Cuota = log(P5100), #Log de pago de cuota
      Ln_Est_arrien = log(P5130), #Log de pago de estimativo de pago de arriendo
      Ln_Pago_arrien = log(P5140)) %>% #Log de pago arriendo
    
    rename(N_cuartos_hog =  P5000,
           Ocup_vivienda = P5090) %>% 
    
    select(-Li,
           -P5010,
           -P5100,
           -P5130,
           -P5140) 
  colnames(test_hogares)
}

### Personas --
{ # Train
  train_personas <- train_personas %>% 
    mutate(
      Mujer = ifelse(P6020==2,1,0), 
      H_Head = ifelse(P6050== 1, 1, 0),#Household head
      Afiliado_SS = ifelse(P6090== 1, 1, 0), #Afiliado a seg social en salud 
      Reg_subs_salud = ifelse(P6100== 3, 1, 0), #Pertenece al regimen de salud subsidiado
      Menor = ifelse(P6040<=6,1,0), # Menores
      EducLevel = ifelse(P6210==9,0,P6210), #Replace 9 with 0
      ocupado = ifelse(is.na(Oc),0,1),
      exper_ult_trab = ifelse(is.na(P6426),0,P6426),
      #Sueldo_tot = P6500+P6510s1+P7070,
      Rec_alimento = ifelse(P6590==1,1,0), #Recibio alimentos como parte de pago del salario
      Rec_vivienda = ifelse(P6600==1,1,0),
      #Pago_negocio = P6750/P6760,
      Cot_pension = ifelse(P6920==1|P6920==3,1,0),
      Ing_extra = P7500s1a1+P7500s3a1+P7510s1a1+P7510s2a1+P7510s5a1,
      Rec_subsidio = ifelse(P7510s3==1,1,0),
      Ln_Ingtot = log(Ingtot),
      Ln_Ingtotob = log(Ingtotob),
      Ln_Ing_extra = log(Ing_extra))
  
  train_personas <- train_personas %>% 
    rename(
      Edad = P6040,
      Ocupacion = P6430,
      Nivel_formalidad = P6870)
  
  
  train_personas <- train_personas %>% 
    select(
      id,
      Orden,
      Clase,
      Dominio,
      Estrato1,
      Edad,
      Mujer, 
      H_Head,#Household head
      Afiliado_SS, #Afiliado a seg social en salud 
      Reg_subs_salud, #Pertenece al regimen de salud subsidiado
      Menor, # Menores
      Oficio,
      Ocupacion,
      Nivel_formalidad,
      EducLevel, #Replace 9 with 0
      ocupado,
      exper_ult_trab,
      Rec_alimento, #Recibio alimentos como parte de pago del salario
      Rec_vivienda,
      Cot_pension,
      Ing_extra,
      Rec_subsidio,
      Pet,
      Ln_Ingtot,
      Ln_Ingtotob,
      Ln_Ing_extra,
      Fex_c,
      Depto,
      Fex_dpto)
  
  colnames(train_personas)
}

{ # Test
  test_personas <- test_personas %>% 
    mutate(
      Mujer = ifelse(P6020==2,1,0), 
      H_Head = ifelse(P6050== 1, 1, 0),#Household head
      Afiliado_SS = ifelse(P6090== 1, 1, 0), #Afiliado a seg social en salud 
      Reg_subs_salud = ifelse(P6100== 3, 1, 0), #Pertenece al regimen de salud subsidiado
      Menor = ifelse(P6040<=6,1,0), # Menores
      EducLevel = ifelse(P6210==9,0,P6210), #Replace 9 with 0
      ocupado = ifelse(is.na(Oc),0,1),
      exper_ult_trab = ifelse(is.na(P6426),0,P6426),
      Rec_alimento = ifelse(P6590==1,1,0), #Recibio alimentos como parte de pago del salario
      Rec_vivienda = ifelse(P6600==1,1,0),
      Cot_pension = ifelse(P6920==1|P6920==3,1,0),
      Rec_subsidio = ifelse(P7510s3==1,1,0))
  
  test_personas <- test_personas %>% 
    rename(
      Edad = P6040,
      Ocupacion = P6430,
      Nivel_formalidad = P6870)
  
  test_personas <- test_personas %>% 
    select(
      id,
      Orden,
      Clase,
      Dominio,
      Edad,
      Mujer, 
      H_Head,#Household head
      Afiliado_SS, #Afiliado a seg social en salud 
      Reg_subs_salud, #Pertenece al regimen de salud subsidiado
      Menor, # Menores
      Oficio,
      Ocupacion,
      Nivel_formalidad,
      EducLevel, #Replace 9 with 0
      ocupado,
      exper_ult_trab,
      Rec_alimento, #Recibio alimentos como parte de pago del salario
      Rec_vivienda,
      Cot_pension,
      Rec_subsidio,
      Pet,
      Fex_c,
      Depto,
      Fex_dpto)
  colnames(train_personas)
}

### Nuevas variables de personas a hogar --
{
  #crear variables
  train_personas_nivel_hogar<- train_personas %>% 
    group_by(id) %>% 
    summarize(nmujeres=sum(Mujer,na.rm=TRUE),
              nmenores=sum(Menor,na.rm=TRUE),
              maxEducLevel=max(EducLevel,na.rm=TRUE),
              nocupados=sum(ocupado,na.rm=TRUE))
}

{ # no se como hacer esto
  train_personas_hogar<- train_personas %>% 
    filter(H_Head==1) %>% 
    select(id,mujer,EducLevel,ocupado) %>% 
    rename(H_Head_mujer=mujer,
           H_Head_Educ_level=EducLevel,
           H_Head_ocupado=ocupado) %>% 
    left_join(train_personas_nivel_hogar)
}

#### 4. Identificacion de pobres en muestra ----
## Hogares
table(train_hogares$Pobre)
table(train_hogares$Pobre)[2]/nrow(train_hogares) 

train_hogares <- train_hogares %>% 
  mutate(Pobre_ing=ifelse(Ing_tot_hog<Lp*Npersug,1,0))

table(train_hogares$Pobre_ing)
# En la muestra hay 33.024 personas en condicion de pobreza
# lo que representa el 20% de la muestra.

# Personas - BORRARRRRRRRRR

## Estuve molestando con el pago de arriendo y la discontinuidad en pobreza
## Por si le quieren dar una revisada 
## Creo que es importante estas variables de pago por vivienda
{
# train_hogares_2 <- train_hogares %>% 
#   mutate(ln_Ingpcug=log(Ingpcug),
#          ln_P5130=log(P5130),
#          ln_P5140=log(P5140),
#          ln_Lp=log(Lp))

# ggplot(aes(ln_Ingpcug, ln_P5140, colour = factor(Pobre)), data = train_hogares_2) +
# geom_point(alpha = 0.2) +
# geom_vline(xintercept = train_hogares_2$ln_Lp, colour = "grey", linetype = 2)+
# stat_smooth(method = "loess", se = F) +
# labs(x = "Ingreso", y = "Pago arriendo")
# 
# ggplot(aes(ln_Ingpcug, ln_P5130, colour = factor(Pobre)), data = train_hogares_2) +
#   geom_point(alpha = 0.2) +
#   geom_vline(xintercept = train_hogares_2$ln_Lp, colour = "grey", linetype = 2)+
#   stat_smooth(method = "lm", se = F) +
#   labs(x = "Ingreso", y = "Pago arriendo")
}

#### 5. Missing ----
## Hogares
miss_var_summary(train_hogares) # Preguntar si se puede colapsar en una variable
# Las variables que estan en missing 
# P5100 cuanto pagan por amortizacion (cuota)
# P5140 Cuanto pagan por arriendo
# P5130 No paga nada, pero cuanto pagaria por arriendo
miss_var_summary(test_hogares)
# Hace falta las mismas 

## Personas
# Hay varios missing en personas
print(miss_var_summary(train_personas), n = 80)
gg_miss_var(train_personas)


#### 6. Guardar nueva base de datos ----
### Hogares
# Train
write.csv(train_hogares, file = "train_hogares_new.csv", row.names = FALSE) # guarda un archivo csv
# Test
write.csv(test_hogares, file = "test_hogares_new.csv", row.names = FALSE) # guarda un archivo csv

### Personas
# Train
write.csv(train_personas, file = "train_personas_new.csv", row.names = FALSE) # guarda un archivo csv
# Test
write.csv(test_personas, file = "test_personas_new.csv", row.names = FALSE) # guarda un archivo csv

#### 7. Colapso de la base de datos en train y test
lm()
train_hogares_new <- read.csv("train_hogares_new.csv")
test_hogares_new <- read.csv("test_hogares_new.csv")
train_hogares_new <- read.csv("train_hogares_new.csv")
test_hogares_new <- read.csv("train_hogares_new.csv")

train <- train_hogares_new %>% 
  left_join(train_personas_hogar) %>% 
  select(-id)

test <- test_hogares_new %>% 
  left_join(test_personas_hogar)

