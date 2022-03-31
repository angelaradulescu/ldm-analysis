library(lattice)
library(lme4)
library(lmerTest)
library(psycho)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%

data <- read.csv('https://raw.githubusercontent.com/angelaradulescu/ldm-analysis/main/ProcessedData/Feedback_Processed_CombinedBehavioralEyetrackingData.csv')

# Clean data if needed
# clean_feedback_data <- feedback_data %>% 
#   clean_names() %>% 
#   select('subj', 'iq', 'within_game_trial', 'trial', 'game', 'entropy', 'age_group', 'po_l', 'rt', 'learned_feat', 'correct') %>% 
#   rename('pol' = 'po_l')
# 
# clean_feedback_data$age_group <- as.factor(clean_feedback_data$age_group)

##### Helper function for rescaling data
scale_this <- function(x){
  (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
}

## Select relevant data and rescale ##
model_data <- data %>% 
  select(Subj, Entropy, Age, Adult, WithinGameTrial, Game, LearnedGame, Learned, IQ) %>% 
  mutate(scaled_age = scale_this(Age),
         scaled_iq = scale_this(IQ),
         subject_id = as.factor(Subj))

#######Continunous Age#######

## Basic Mixed Model ## 
## just Age*LearnedFeat as fixed effects Subj as random effect. ##
m1 <- lmer(data = model_data, formula = Entropy ~ Age*LearnedGame + (1|subject_id))
summary(m1)

## Omnibus test. 
car::Anova(m1, type = '3')

## More Mixed Model ## 
## Age*WithinGameTrial*Game*LearnedFeat*IQ as fixed effects Subj as random effect. ##

# fit warnings: Some predictor variables are on very different scales: consider rescaling # NB: no longer appearing after rescaling age and iq

m2 <- lmer(data = model_data, formula = Entropy ~ scaled_age*WithinGameTrial*Game*LearnedGame*scaled_iq + (1|subject_id))
summary(m2)

## Omnibus test. 
car::Anova(m2, type = '3')


## Even More Mixed Model ## 
## Age*WithinGameTrial*Game*LearnedFeat*IQ as fixed effects that act on Subj as random effect. ##
## Fitting this model resulted in singularity of the model. FMI: enter ?isSingular in console

m3 <- lmer(data = model_data, formula = Entropy ~ scaled_age*WithinGameTrial*Game*LearnedGame*scaled_iq + ((WithinGameTrial * (Game+LearnedGame))|subject_id))
summary(m3)

## Omnibus test. 
car::Anova(m3, type = '3')

### Mixed Model Playground (trying out different things) ###

m3 <- lmer(data = data, formula = Entropy ~ Age*(WithinGameTrial + Game) + ((WithinGameTrial + Game)|Subj))
summary(m3)

## Omnibus test. 
car::Anova(m3, type = '3')

