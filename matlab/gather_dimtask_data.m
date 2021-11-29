function gather_dimtask_data(dataset)
% Gathers Dimensions Task data for Learning and Decision Making across
% development project
% 
% dataset=1: LDM dataset collected by Gail Rosenbaum
% BOX link: https://nyu.app.box.com/folder/39621422799 

% Handle discrepancies between datasets
data_specs = set_specs(dataset);

% Get number of subjects
nSubs = length(data_specs.subj);

% Combine data across subjects
for s = 1:nSubs
    %% Set paths
    subjID = num2str(data_specs.subj(s));
    AllData{s,1} = data_specs.subj(s);
    fprintf(sprintf('Subject %d \n',data_specs.subj(s)));
    
    f = dir(strcat(data_specs.behav,strcat('Subj',num2str(subjID),'*')));
    behav_path = fullfile(f.folder, f.name);
    
    %% Load files
    fprintf('Loading behavioral data... \n');
    load(behav_path); 
     
    %% Initialize variables 
    
    % Behavior
    c.b.Stimuli = [];
    c.b.RowOrder = [];
    c.b.StimOn = [];
    c.b.Chosen = [];
    c.b.chosenColumn = [];
    c.b.RT = [];
    c.b.Outcome = [];
    c.b.OutcomeOn = [];
    c.b.CorrectFeature = [];
    c.b.CorrectDim = [];
    c.b.Correct = [];
    c.b.NGames = 0;
    c.b.GameNumber = [];
    c.b.GameFB = [];
    c.b.centerDim = [];
    c.b.centerFeat = [];
    c.b.displayNum = [];
    c.b.dimSaliency = [];
    c.b.stimSaliency = [];
    c.b.pointOfLearning = [];
     
    %% Loop over runs 
    for r = 1:data_specs.nRuns
     
        %% Subset run data
        runStr = sprintf('Run%i',r);
        behavData = Data{1,r};
       
        %% Add behavior
        c.b.Stimuli = [c.b.Stimuli; behavData.Stimuli];
        c.b.RowOrder = [c.b.RowOrder; behavData.RowOrder]; 
        c.b.StimOn = [c.b.StimOn; behavData.StimOn];
        c.b.Chosen = [c.b.Chosen; behavData.Chosen];
        cc = getChosenCol(behavData); c.b.chosenColumn = [c.b.chosenColumn; cc];
        c.b.RT = [c.b.RT; behavData.RT];
        c.b.Outcome = [c.b.Outcome; behavData.Outcome];
        c.b.OutcomeOn = [c.b.OutcomeOn; behavData.OutcomeOn];
        c.b.CorrectFeature = [c.b.CorrectFeature; behavData.CorrectFeature];
        c.b.CorrectDim = [c.b.CorrectDim; behavData.CorrectDim];
        if (data_specs.subj(s) ~= 603) % subject 603 in dataset 1 missing this field 
            c.b.Correct = [c.b.Correct; behavData.Correct];
        end
        c.b.NGames = c.b.NGames + behavData.NGames;
        c.b.GameNumber =  [c.b.GameNumber; behavData.GameNumber];
              
    end
    
    % Record combined structures and task parameters
    AllData{s,2} = c;
    if dataset < 3 
        AllData{s,3} = Parms;
    else
        AllData{s,3} = params;
    end
    
end

outputpath = fullfile(data_specs.output,'AllData.mat');
save(outputpath,'AllData');

end

%% Helpers
function[data_specs] = set_specs(dataset)

switch dataset
            
    case 1 % first LDM experiment
        
        data_specs.behav = '/Users/angelaradulescu/Dropbox/NYU/Research/LDM/RawData/';
        data_specs.output = '/Users/angelaradulescu/Dropbox/NYU/Research/LDM/ProcessedData/';    
        data_specs.subj = [23, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 53, 54, 55, 57, 58, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71];
        data_specs.nRuns = 10;
        
end

end

function [cc] = getChosenCol(behavData)
% Determine chosen column for each trial (spatially resolved)

nTrials = length(behavData.RT);
cc = NaN(nTrials,1);
for t = 1:nTrials
    if ~isnan(behavData.Chosen(t,1)) % this is a valid trial
        cc(t) = find(([1;1;1]*behavData.Chosen(t,:)) == squeeze(behavData.Stimuli(t,:,:)),1);
    end
end
 
end
