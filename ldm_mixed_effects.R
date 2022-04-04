library(lattice)
library(lme4)
library(lmerTest)
library(psycho)
library(sjPlot)  # for plotting lmer and glmer mods
library(emmeans) # for post-hoc tests

getwd()

data <- read.csv('/Users/angelaradulescu/Dropbox/NYU/Research/LDM/ldm-analysis/ProcessedData/Feedback_Processed_CombinedBehavioralEyetrackingData.csv')
head(data, 10)

## Model. 
m1 <- lmer(data = data, formula = Entropy ~ Age*LearnedFeat + (1|Subj))
summary(m1)

## Omnibus test. 
car::Anova(m1, type = '3')

## Plot effects.
sjPlot::plot_model(m1, type = 'int')

