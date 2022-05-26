library(lattice)
library(lme4)
library(lmerTest)
library(psycho)
library(sjPlot)  # for plotting lmer and glmer mods
library(emmeans) # for post-hoc tests
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%

##################### Import Data #####################
data <- read.csv('https://raw.githubusercontent.com/angelaradulescu/ldm-analysis/main/ProcessedData/Feedback_Processed_CombinedBehavioralEyetrackingData.csv')
head(data, 10)

# Clean data if needed
# clean_feedback_data <- feedback_data %>% 
#   clean_names() %>% 
#   select('subj', 'iq', 'within_game_trial', 'trial', 'game', 'entropy', 'age_group', 'po_l', 'rt', 'learned_feat', 'correct') %>% 
#   rename('pol' = 'po_l')
# 
# clean_feedback_data$age_group <- as.factor(clean_feedback_data$age_group)

##################### Helper function for rescaling data ##################### 
scale_this <- function(x){
  (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
}

##################### Select relevant data and rescale #####################
model_data <- data %>% 
  select(Subj, Entropy, Age, AgeGroup, WithinGameTrial, Game, LearnedGame, Learned, IQ, PoL) %>% 
  mutate(scaled_age = scale_this(Age),
         scaled_iq = scale_this(IQ),
         subject_id = as.factor(Subj),
         age_group = as.factor(AgeGroup))


# make new AlignedTrial value and LearnedYet value containing whether trial is pre- or post- PoL
# learning_trials <- model_data[model_data$LearnedGame == 'True',]
# learning_trials$AlignedTrial = learning_trials$WithinGameTrial - learning_trials$PoL
model_data$AlignedTrial = model_data$WithinGameTrial - model_data$PoL
model_data$LearnedYet = model_data$AlignedTrial > 0
model_data %<>% mutate(learned_yet = as.factor(LearnedYet))

#####################Continunous Age + Game-wise Learned Parameter#####################

## Basic Mixed Model ## 
## just Age*LearnedGame as fixed effects Subj as random effect. ##
m1 <- lmer(data = model_data, formula = Entropy ~ scaled_age*LearnedGame + (1|subject_id))
summary(m1)

m1c <- lmer(data = model_data, formula = Entropy ~ AgeGroup*WithinGameTrial*LearnedGame + (1|subject_id))
summary(m1c)

m1d <- lmer(data = model_data, formula = Entropy ~ AgeGroup + (1|subject_id))
summary(m1d)

## Omnibus test. 
car::Anova(m1, type = '3')


## More Mixed Model ## 
## Age*WithinGameTrial*Game*LearnedGame as fixed effects Subj as random effect. ##

# fit warnings: Some predictor variables are on very different scales: consider rescaling # NB: no longer appearing after rescaling age and iq

m2 <- lmer(data = model_data, formula = Entropy ~ scaled_age*WithinGameTrial*Game*LearnedGame + (1|subject_id))
summary(m2)

## Omnibus test. 
car::Anova(m2, type = '3')


## Even More Mixed Model ## 
## Age*WithinGameTrial*Game*LearnedGame as fixed effects that act on Subj as random effect. ##
## Fitting this model resulted in singularity of the model. FMI: enter ?isSingular in console

m3 <- lmer(data = model_data, formula = Entropy ~ scaled_age*WithinGameTrial*Game*LearnedGame + ((WithinGameTrial * (Game+LearnedGame))|subject_id))
summary(m3)

## Omnibus test. 
car::Anova(m3, type = '3')

### Mixed Model Playground (trying out different things) ###

m4 <- lmer(data = model_data, formula = Entropy ~ scaled_age*(WithinGameTrial + Game) + ((WithinGameTrial + Game)|subject_id))
summary(m4)

## Omnibus test. 
car::Anova(m4, type = '3')

#####################Continunous Age, separated by PoL##########################

## Basic Mixed Model ## 
## just Age as fixed effect and Subj as random effect. ##
m1 <- lmer(data = model_data, formula = Entropy ~ scaled_age*LearnedYet + (1|subject_id))
summary(m1)

## Omnibus test. 
car::Anova(m1, type = '3')

## More Mixed Model ## 
## Age*AlignedTrial*Game*LearnedGame as fixed effects Subj as random effect. ##

m2 <- lmer(data = model_data, formula = Entropy ~ scaled_age*AlignedTrial*Game*LearnedGame + (1|subject_id))
summary(m2)

## Omnibus test. 
car::Anova(m2, type = '3')


## Even More Mixed Model ## 
## Age*AlignedTrial*Game*LearnedYet as fixed effects that act on Subj as random effect. ##
## Fitting this model resulted in singularity of the model. FMI: enter ?isSingular in console

m3 <- lmer(data = model_data, formula = Entropy ~ scaled_age*AlignedTrial*Game*LearnedYet + ((AlignedTrial * (Game+LearnedYet))|subject_id))
summary(m3)

## Omnibus test. 
car::Anova(m3, type = '3')

### Mixed Model Playground (trying out different things) ###

m4 <- lmer(data = model_data, formula = Entropy ~ scaled_age*(WithinGameTrial + Game) + ((WithinGameTrial + Game)|subject_id))
summary(m4)

## Omnibus test. 
car::Anova(m4, type = '3')


############################ Categorical Age ###################################

m1a <- lmer(data = model_data, formula = Entropy ~ AgeGroup*LearnedGame + (1|subject_id/LearnedGame))
summary(m1a)

m1b <- lmer(data = model_data, formula = Entropy ~ AgeGroup*WithinGameTrial + (1|subject_id))
summary(m1b)

## Omnibus test. 
car::Anova(m1, type = '3')

## Plot effects.
sjPlot::plot_model(m1, type = 'int')

## Basic Mixed Model ## 
## just Age*LearnedGame as fixed effects Subj as random effect. ##
m1 <- lmer(data = model_data, formula = Entropy ~ age_group*LearnedGame + (1|subject_id))
summary(m1)

## Omnibus test. 
car::Anova(m1, type = '3')

## More Mixed Model ## 
## Age*WithinGameTrial*Game*LearnedGame as fixed effects Subj as random effect. ##

m2 <- lmer(data = model_data, formula = Entropy ~ age_group*WithinGameTrial*Game*LearnedGame + (1|subject_id))
summary(m2)

## Omnibus test. 
car::Anova(m2, type = '3')

##################### Categorical Age, separated by PoL##########################

## Basic Mixed Model ## 
## just Age as fixed effect and Subj as random effect. ##
m1 <- lmer(data = model_data, formula = Entropy ~ age_group*learned_yet + (1|subject_id))
summary(m1)

## Omnibus test. 
car::Anova(m1, type = '3')

## More Mixed Model ## 
## Age*WithinGameTrial*Game*LearnedGame as fixed effects Subj as random effect, with WithinGameTrial nested##

m2 <- lmer(data = model_data, formula = Entropy ~ age_group*WithinGameTrial*Game*learned_yet + (1|subject_id/WithinGameTrial))
summary(m2)

## Omnibus test. 
car::Anova(m2, type = '3')

## Plot effects.
sjPlot::plot_model(m2, type = 'int')

equation1=function(x){coef(m2)[2]*x+coef(m2)[1]}
equation2=function(x){coef(m2)[2]*x+coef(m2)[1]+coef(m2)[3]}

ggplot(model_data,aes(y=Entropy,x=WithinGameTrial,color=AgeGroup))+geom_point()+
  stat_function(fun=equation1,geom="line",color=scales::hue_pal()(2)[1])+
  stat_function(fun=equation2,geom="line",color=scales::hue_pal()(2)[2])

## Even More Mixed Model ## 
## Age*AlignedTrial*Game*LearnedYet as fixed effects that act on Subj as random effect. ##
## Fitting this model resulted in singularity of the model. FMI: enter ?isSingular in console

m3 <- lmer(data = model_data, formula = Entropy ~ scaled_age*AlignedTrial*Game*LearnedYet + ((AlignedTrial * (Game+LearnedYet))|subject_id))
summary(m3)

## Omnibus test. 
car::Anova(m3, type = '3')

### Mixed Model Playground (trying out different things) ###

m4 <- lmer(data = model_data, formula = Entropy ~ scaled_age*(WithinGameTrial + Game) + ((WithinGameTrial + Game)|subject_id))
summary(m4)

## Omnibus test. 
car::Anova(m4, type = '3')


############### DUMP #############################################

########## Making Separate Models ################
# split data based on pre and post learning
pre_learning_trials <- model_data[model_data$AlignedTrial < 0,]
post_learning_trials <- model_data[model_data$AlignedTrial > 0,]

# pre learning
pre_m1 <- lmer(data = pre_learning_trials, formula = Entropy ~ scaled_age + (1|subject_id))
summary(pre_m1)

## Omnibus test. 
car::Anova(pre_m1, type = '3')

# post learning
post_m1 <- lmer(data = post_learning_trials, formula = Entropy ~ scaled_age + (1|subject_id))
summary(post_m1)

## Omnibus test. 
car::Anova(post_m1, type = '3')


