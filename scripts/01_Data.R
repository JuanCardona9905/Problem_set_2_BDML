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
       wd <- "/Users/camilabeltran/OneDrive/Educación/PEG - Uniandes/BDML/Problem_set_2_BDML",
       ifelse(grepl("Juan",getwd()),
              wd <- "C:/Users/Juan/Documents/Problem_set_2",
              ifelse(grepl("juanp.rodriguez",getwd()),
                     wd <- "C:/Users/juanp.rodriguez/Documents/GitHub/Problem_set_1",
                     ifelse(grepl("C:/Users/User",getwd()),
                            wd <- "C:/Users/User/OneDrive - Universidad de los andes/Big Data y Machine Learning/Problem_set_1/Problem_set_1",
                            ifelse(grepl("/Users/aleja/",getwd()),
                                   wd <- "/Users/aleja/Documents/Maestría Uniandes/Clases/Big Data y Machine Learning/Repositorios Git Hub/Problem_set_1)",
                                   wd <- "otro_directorio")))))
}

#### 2. Importar bases de datos ----
{
### Importar las bases de entrenamiento 
# Personas
setwd(paste0(wd,"/data"))
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

#### 3. Modificacion base de datos ----
# Dejar solo las variables que se comparten en train y test.
{
##personas
train_personas <- train_personas[,c(colnames(test_personas))]
##hogares - se dejan además las variables que queremos predecir: Pobre e ingreso del hogar
train_hogares <- train_hogares[,c(colnames(test_hogares),"Pobre","Ingtotug","Ingtotugarr","Ingpcug")]
}
### Personas --
{ 
pre_process_personas <- function(data,...){
  data <- data %>% 
    
    #modificar variables
    mutate(
      Mujer = ifelse(P6020==2,1,0), 
      H_Head = ifelse(P6050== 1, 1, 0),#Household head
      H_Head_mujer = ifelse(P6050== 1&P6020==2, 1, 0), #Hoysehold head women
      Afiliado_SS = ifelse(P6090== 1, 1, 0), #Afiliado a seg social en salud
      Afiliado_SS = ifelse(is.na(Afiliado_SS),0,Afiliado_SS), 
      Reg_subs_salud = ifelse(P6100== 3, 1, 0), #Pertenece al regimen de salud subsidiado
      Reg_subs_salud = ifelse(is.na(Reg_subs_salud),0,Reg_subs_salud), 
      Menor_5 = ifelse(P6040<6,1,0), # Menores de 5 años
      Menor_6_11 = ifelse(P6040>5 & P6040<12,1,0), # Menores entre 6 y 11 años
      Menor_12_17 = ifelse(P6040>11 & P6040<18,1,0), # Menores entre 12 y 17 años
      EducLevel = ifelse(P6210==9,0,P6210), #Replace 9 with 0
      ocupado = ifelse(is.na(Oc),0,1),
      exper_ult_trab = ifelse(is.na(P6426),0,P6426),
      #Sueldo_tot = P6500+P6510s1+P7070,
      Rec_alimento = ifelse(P6590==1,1,0), #Recibio alimentos como parte de pago del salario
      Rec_alimento = ifelse(is.na(Rec_alimento),0,Rec_alimento),
      Rec_vivienda = ifelse(P6600==1,1,0), #Recibio vivienda como parte de pago del salario
      Rec_vivienda = ifelse(is.na(Rec_vivienda),0,Rec_vivienda), 
      #Pago_negocio = P6750/P6760,
      Cot_pension = ifelse(P6920==1|P6920==3,1,0),
      Cot_pension = ifelse(is.na(Cot_pension),0,Cot_pension),
      #Ing_extra = P7500s1a1+P7500s3a1+P7510s1a1+P7510s2a1+P7510s5a1,
      Rec_subsidio = ifelse(P7510s3==1,1,0),
      Rec_subsidio = ifelse(is.na(Rec_subsidio),0,Rec_subsidio),
      P6870 = ifelse(is.na(P6870),0,P6870))%>% #pone 0 en NA para nivel de formalidad
      #Ln_Ingtot = log(Ingtot),
      #Ln_Ingtotob = log(Ingtotob),
      #Ln_Ing_extra = log(Ing_extra))
    
    #modificar nombres
    rename(
      Edad = P6040,
      Ocupacion = P6430,
      Nivel_formalidad = P6870)
    }  
}  

train_personas <- pre_process_personas(train_personas)
test_personas <- pre_process_personas(test_personas)

{#crear bases de personas a nivel hogar
personas_nivel_hogar <- function(data,...){
  data <- data %>% 
    group_by(id)%>%
    summarize(nmujeres=sum(Mujer,na.rm=TRUE),
              nmenores_5=sum(Menor_5,na.rm=TRUE),
              nmenores_6_11=sum(Menor_6_11,na.rm=TRUE),
              nmenores_12_17=sum(Menor_12_17,na.rm=TRUE),
              maxEducLevel=max(EducLevel,na.rm=TRUE),
              nocupados=sum(ocupado,na.rm=TRUE))
}

train_personas_hogar <- personas_nivel_hogar(train_personas)
test_personas_hogar <- personas_nivel_hogar(test_personas)  

#agregar variables de jefe de hogar
personas_jefe_hogar <- function(data,...){
   data <- data %>% 
      filter(H_Head==1) %>% 
      select(id,Mujer,EducLevel,ocupado,Afiliado_SS,Reg_subs_salud,exper_ult_trab,Rec_alimento,Rec_vivienda,Cot_pension,Rec_subsidio,Nivel_formalidad)%>% 
      rename(Head_Mujer=Mujer,
             Head_EducLevel=EducLevel,
             Head_ocupado=ocupado,
             Head_Afiliado_SS=Afiliado_SS,
             Head_Reg_subs_salud=Reg_subs_salud,
             Head_exper_ult_trab=exper_ult_trab,
             Head_Rec_alimento=Rec_alimento,
             Head_Rec_vivienda=Rec_vivienda,
             Head_Cot_pension=Cot_pension,
             Head_Rec_subsidio=Rec_subsidio,
             Head_Nivel_formalidad=Nivel_formalidad)
   }
  
train_personas_hogar <- left_join(train_personas_hogar,personas_jefe_hogar(train_personas),by = "id")
test_personas_hogar <- left_join(test_personas_hogar,personas_jefe_hogar(test_personas),by = "id")
}

### Hogares --
{ #Agregar variables de personas a la base de Hogares
train_hogares <- left_join(train_hogares,train_personas_hogar,by="id")
test_hogares <- left_join(test_hogares,test_personas_hogar,by="id")
}

{ #Cambios en train y test
pre_process_hogares <- function(data,...){
      data <- data %>% 
      
      #modificar variables
      mutate(
      Dominio=factor(Dominio),
      maxEducLevel=factor(maxEducLevel,levels=c(0:6), labels=c("Ns",'Ninguno', 'Preescolar','Primaria', 'Secundaria','Media', 'Universitaria')),
      Cabecera = ifelse(Clase==1,1,0), 
      DormitorXpersona = P5010/Nper,
      P5140 = ifelse(is.na(P5140),P5130,P5140), #pone valores de pago estimado de arriendo a NA 
      P5100 = ifelse(is.na(P5100),0,P5100), #pone 0 en NA (valor cuota)
      Ln_Cuota = log(P5100), #Log de pago de cuota
      #Ln_Est_arrien = log(P5130), #Log de pago de estimativo de pago de arriendo
      Ln_Pago_arrien = log(P5140)) %>% #Log de pago arriendo 
    
      #renombrar variables 
      rename(N_cuartos_hog =  P5000,
             Ocup_vivienda = P5090) %>%
        
      select(-P5130) #quita la variable de cuanto pagaría por arriendo xq solo se utiliza para la imputacion
}

train_hogares <- pre_process_hogares(train_hogares)
test_hogares <- pre_process_hogares(test_hogares)

}

{ #Cambios en variables de predicción 
train_hogares <- train_hogares %>% 
  mutate(
    Pobre=factor(Pobre,levels=c(0,1),labels=c("No","Yes")),#pobre como factor
    Ln_Ing_tot_hogar = log(Ingtotug),
    Ln_Ing_tot_hogar_imp_arr = log(Ingtotugarr),
    Ln_Ing_tot_hogar_per_cap = log(Ingpcug))
}

#### 4. Identificacion de pobres en muestra ----
## Hogares
table(train_hogares$Pobre)
table(train_hogares$Pobre)[2]/nrow(train_hogares) 
# En la muestra hay 33.024 personas en condicion de pobreza
# lo que representa el 20% de la muestra.

#### 5. Missing ----
miss_var_summary(train_hogares) #sin missings

#### 6. Guardar nueva base de datos --
rm(list = c("test_personas_hogar","train_personas_hogar",
            "personas_jefe_hogar","personas_nivel_hogar",
            "pre_process_hogares","pre_process_personas"))

save.image("base_final.RData")

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
