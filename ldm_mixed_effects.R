library(lattice)
library(lme4)
library(lmerTest)
library(psycho)

getwd()

data <- read.csv('/Users/angelaradulescu/Dropbox/NYU/Research/LDM/ldm-analysis/ProcessedData/Feedback_Processed_CombinedBehavioralEyetrackingData.csv')
print(data)

## Model. 
m1 <- lmer(data = data, formula = Entropy ~ Age*LearnedFeat + (1|Subj))
summary(m1)

## Omnibus test. 
car::Anova(m1, type = '3')


