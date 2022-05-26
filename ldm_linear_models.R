## Linear Regression Model ## 
# Load needed libraries
library(tidyverse)
library(glue)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(afex)
library(knitr)
library(kableExtra)
library(broom)
library(lattice)
library(lme4)
library(lmerTest)
library(psycho)
library(dplyr)    # alternatively, this also loads %>%
require(ggiraph)
require(ggiraphExtra)
require(plyr)

# scale function (z-score)
scale_this <- function(x){
  (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
}

# run wasi iq analysis
age_iq_data <- read.csv('https://raw.githubusercontent.com/angelaradulescu/ldm-analysis/main/ProcessedData/ageIQMap.csv')

age_iq_data %<>%
  mutate(age_scaled = scale_this(Age),
         iq_scaled = scale_this(IQ))

lm(IQ ~ Age, age_iq_data) %>%
  tidy() %>%
  kable()%>%
  kable_classic(full_width = F, html_font = "Arial") %>%
  kable_styling("striped", position = "float_left")

####################### AgexNumberOfGamesLearned ###############################
behav_data <- read.csv('https://raw.githubusercontent.com/angelaradulescu/ldm-analysis/main/ProcessedData/CleanedProcessedBehavioralData.csv')

sub_summary <- behav_data %>% 
  select(Subj, Age, LearnedGame, PoL) %>%
  unique() %>% 
  group_by(Subj,Age) %>% 
  mutate(LearnedGame = as.logical(LearnedGame)) %>% 
  summarize(GamesLearned = sum(LearnedGame))
  
sub_summary$age_scaled <- scale_this(sub_summary$Age)

lm(GamesLearned ~ age_scaled, sub_summary) %>%
  tidy() %>%
  kable()%>%
  kable_classic(full_width = F, html_font = "Arial") %>%
  kable_styling("striped", position = "float_left")

###################### Visualize Feedback Trial Entropy ########################
data <- read.csv('https://raw.githubusercontent.com/angelaradulescu/ldm-analysis/main/ProcessedData/Feedback_Processed_CombinedBehavioralEyetrackingData.csv')

model_data <- data %>% 
  select(Subj, Entropy, Age, Adult, AgeGroup, WithinGameTrial, Game, LearnedGame, Learned, IQ, PoL) %>% 
  mutate(scaled_age = scale_this(Age),
         scaled_iq = scale_this(IQ),
         subject_id = as.factor(Subj))
model_data$AlignedTrial <- model_data$WithinGameTrial - model_data$PoL
model_data$LearnedYet <- model_data$AlignedTrial > 0
adult_data <- model_data[model_data$Adult == 'True',]
adole_data <- model_data[model_data$Adult == 'False',]

### Fit 2 predictor variables ###

## Fit Trial and Age Group without interaction ##
fit = lm(Entropy~LearnedGame*AgeGroup,data=model_data)
summary(fit)

equation1=function(x){coef(fit)[2]*x+coef(fit)[1]}
equation2=function(x){coef(fit)[2]*x+coef(fit)[1]+coef(fit)[3]}

ggplot(model_data,aes(y=Entropy,x=WithinGameTrial,color=AgeGroup))+geom_point()+
  stat_function(fun=equation1,geom="line",color=scales::hue_pal()(2)[1])+
  stat_function(fun=equation2,geom="line",color=scales::hue_pal()(2)[2])

ggPredict(fit,se=TRUE,interactive=TRUE)

## Fit Trial and Learned Game with interaction ##
fit1 = lm(Entropy~WithinGameTrial*LearnedGame,data=model_data)
summary(fit1)

equation1_1=function(x){coef(fit1)[2]*x+coef(fit1)[1]}
equation2_1=function(x){coef(fit1)[2]*x+coef(fit1)[1]+coef(fit1)[3]}

ggplot(model_data,aes(y=Entropy,x=WithinGameTrial,color=LearnedGame))+geom_point()+
  stat_function(fun=equation1_1,geom="line",color=scales::hue_pal()(2)[1])+
  stat_function(fun=equation2_1,geom="line",color=scales::hue_pal()(2)[2])

ggPredict(fit1,se=TRUE,interactive=TRUE)

### Fit 3 predictor variables, with and without interaction ##
fit2 = lm(Entropy~WithinGameTrial*LearnedGame*AgeGroup,data=model_data)
summary(fit2)

equation1_2=function(x){coef(fit2)[2]*x+coef(fit2)[1]}
equation2_2=function(x){coef(fit2)[2]*x+coef(fit2)[1]+coef(fit2)[3]}

ggplot(model_data,aes(y=Entropy,x=WithinGameTrial,color=AgeGroup,linetype=LearnedGame))+geom_point()+
  stat_function(fun=equation1_2,geom="line",color=scales::hue_pal()(2)[1])+
  stat_function(fun=equation2_2,geom="line",color=scales::hue_pal()(2)[2])

ggPredict(fit2,se=TRUE,interactive=TRUE)

##################### 

