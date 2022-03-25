library(lattice)
library(lme4)
library(psycho)

getwd()

data <- read.csv('/Users/angelaradulescu/Dropbox/NYU/Research/LDM/ldm-analysis/ProcessedData/Feedback_Processed_CombinedBehavioralEyetrackingData.csv')
print(data)

## Model. 
m1 <- lmer(data = data, formula = Entropy ~ Age*LearnedFeat + (1|Subj))

results <- analyze(m1)
summary(m1)
anova(m1)


