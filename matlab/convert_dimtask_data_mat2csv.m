function [] = convert_dimtask_data_mat2csv(dataset)
% Converts Dimensions Task data for Learning and Decision Making across
% development project from mat to csv
% 
% dataset=1: LDM dataset collected by Gail Rosenbaum
% BOX link: https://nyu.app.box.com/folder/39621422799


% Handle discrepancies between datasets
data_specs = set_specs(dataset);

% Load data
data_path = fullfile(data_specs.path,'AllData.mat');
load(data_path);
Data = AllData;

% Get number of subjects.
nSubj = length(Data);

% Initialize arrays for variables that we want to extract: 
subject = [];       % subject 
trial = [];         % trial 
gameNumber = [];    % game number
stimuli = [];       % first 3 are stimulus 1, middle 3 are stimulus 2 and last 3 are stimulus 3, coding in feature space (1-9)
choice = [];        % which stimulus was chosen (1-3) 
chosen = [];        % which features were chosen (1-9) 
unchosen = [];      % which features were not chosen (1-9, grouped in 3s)
outcome = [];       % outcome (1/0)
relDim = [];        % relevant dimension (1-3) 
relFeat = [];       % relevant feature (1-9)
centerDim = [];     % center dimension (1-3)
centerFeat = [];    % center feature (1-9)
correct = [];       % correct or incorrect choice
learned_feat = [];  % feature reported at end of game 
pol = [];           % point of learning (trial on which participants consistently chose target feature on every remaining trial)
rt = [];            % reaction time

% Loop over subjects
for s = 1:nSubj 
    
%     subjID = data_specs.subj(s);
%     display(subjID)
    
    nTrials = length(Data{1,2}.b.Outcome);  

    thisSubject = repmat(Data{s,1},nTrials,1);
    
    thisTrial = (1:nTrials)';
    thisRelDim = Data{s,2}.b.CorrectDim;
    
    thisCenterDim = Data{s,2}.b.centerDim;
    [a, b] = size(thisCenterDim);
    if (a == 0)
        thisCenterDim = NaN(nTrials,1);
    end

    thisCenterFeat = Data{s,2}.b.centerFeat;  
    [a, b] = size(thisCenterFeat);
    if (a == 0)
        thisCenterFeat = NaN(nTrials,1);
    end

    thisCorrect = Data{s,2}.b.Correct;
    [a, b] = size(thisCorrect);
    if (a == 0)
        thisCorrect = NaN(nTrials,1);
    end
    % thisCorrect(isnan(thisCorrect)) = 0;

    % Mark game number based on GameNumber switches
    thisGameNumber = NaN(nTrials,1);
    gn = 1;
    hlp  = [diff(Data{s,2}.b.GameNumber); 0]; 
    for t = 1:nTrials
        thisGameNumber(t) = gn;
        if hlp(t)
           gn = gn+1; 
        end
    end

    % Mark game number based on dimension switches 
    % The below only works assuming the relevant dimension always changes between games
    % This is not robust to datasets 1 and 2, but keeping commented out as legacy  
    % thisGameNumber = NaN(nTrials,1);
    % gn = 1;
    % hlp  = [diff(thisRelDim); 0]; 
    % for t = 1:nTrials
    %     thisGameNumber(t) = gn;
    %     if hlp(t)
    %        gn = gn+1; 
    %     end
    % end
    
    % Mark feature that participants have learned 
    thisGameFB = Data{s,2}.b.GameFB;
    thisLearnedFeat = NaN(nTrials,1);
    % thisLearnedFeat(find(thisGameFB(:,1) == 1)) = thisGameFB(find(thisGameFB(:,1) == 1),2) + 0;
    % thisLearnedFeat(find(thisGameFB(:,1) == 2)) = thisGameFB(find(thisGameFB(:,1) == 2),2) + 3;
    % thisLearnedFeat(find(thisGameFB(:,1) == 3)) = thisGameFB(find(thisGameFB(:,1) == 3),2) + 6;
  
    % Mark point of learning 
    thisPointOfLearning = NaN(nTrials,1);
    % size(thisPointOfLearning)
    % nGames = max(thisGameNumber);
    % for g = 1:nGames
    %   gameTrials = find(thisGameNumber == g);
    %   gameCorrect = thisCorrect(gameTrials);
    %   gameMistakes = find(gameCorrect==0);
    %   % Handle case in which there were only correct trials
    %   if numel(gameMistakes)
    %       thisPointOfLearning(gameTrials) = gameMistakes(end); 
    %   else
    %       thisPointOfLearning(gameTrials) = 1;
    %   end
    % end
    
    thisRelFeat = Data{s,2}.b.CorrectFeature + 3*(Data{s,2}.b.CorrectDim-1);
    thisChosen = Data{s,2}.b.Chosen + repmat([0 3 6],nTrials,1); 
    thisOutcome = Data{s,2}.b.Outcome;
    thisRT = Data{s,2}.b.RT;

    Stim_hlp = Data{s,2}.b.Stimuli;
    Stim_hlp(nTrials+1:end,:,:) = [];
    Stim_hlp(:,:,2) = Stim_hlp(:,:,2)+3;
    Stim_hlp(:,:,3) = Stim_hlp(:,:,3)+6;

    theseStimuli = NaN(nTrials,9);
    thisChoice = NaN(nTrials,1);
    thisUnchosen = NaN(nTrials,6);

    % Mark chosen stimulus 
    for t = 1:nTrials 
        theseStimuli(t,:) = reshape(squeeze(Stim_hlp(t,:,:))',1,9);
        if ~isnan(thisChosen(t,1)) 
            thisChoice(t) = find(([1;1;1]*thisChosen(t,:)) == squeeze(Stim_hlp(t,:,:)),1);
            f_help = [1:9]; f_help(thisChosen(t,:)) = [];
            thisUnchosen(t,:) = f_help;
        end
    end
    
    % Append to arrays 
    subject = [subject; thisSubject];
    trial = [trial; thisTrial];
    gameNumber = [gameNumber; thisGameNumber];
    stimuli = [stimuli; theseStimuli];
    choice = [choice; thisChoice];
    chosen = [chosen; thisChosen];
    unchosen = [unchosen; thisUnchosen];
    outcome = [outcome; thisOutcome];
    relDim = [relDim; thisRelDim];
    relFeat = [relFeat; thisRelFeat];
    centerDim = [centerDim; thisCenterDim];
    centerFeat = [centerFeat; thisCenterFeat];
    correct = [correct; thisCorrect];
    learned_feat = [learned_feat; thisLearnedFeat];
    pol = [pol; thisPointOfLearning];
    rt = [rt; thisRT]; 
        
end

%% Gather into single array and write to csv
dat = [subject trial gameNumber stimuli choice chosen unchosen outcome relDim relFeat centerDim centerFeat correct learned_feat pol rt];

headers = {'Subj','Trial','Game',...
            'Stim11','Stim12','Stim13','Stim21','Stim22','Stim23','Stim31','Stim32','Stim33'...
                'Choice','Chosen1','Chosen2','Chosen3',...
                    'Unchosen1','Unchosen2','Unchosen3','Unchosen4','Unchosen5','Unchosen6',...
                        'Outcome','Dim','Feat','CenterDim','CenterFeat','Correct','LearnedFeat','PoL','RT'};
    
filename = fullfile(data_specs.path,'AllData.csv');
csvwrite_with_headers(filename,dat,headers);

end

function[data_specs] = set_specs(dataset)

switch dataset
            
    case 1 % first LDM experiment
        
        data_specs.path  = '/Users/angelaradulescu/Dropbox/NYU/Research/LDM/ProcessedData/';
        data_specs.subj = [23, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 53, 54, 55, 57, 58, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71];
        data_specs.nRuns = 10;
        
end

end

function csvwrite_with_headers(filename,m,headers,r,c)

% This function functions like the build in MATLAB function csvwrite but
% allows a row of headers to be easily inserted
%
% known limitations
% 	The same limitation that apply to the data structure that exist with 
%   csvwrite apply in this function, notably:
%       m must not be a cell array
%
% Inputs
%   
%   filename    - Output filename
%   m           - array of data
%   headers     - a cell array of strings containing the column headers. 
%                 The length must be the same as the number of columns in m.
%   r           - row offset of the data (optional parameter)
%   c           - column offset of the data (optional parameter)
%
%
% Outputs
%   None

%% initial checks on the inputs
if ~ischar(filename)
    error('FILENAME must be a string');
end

% the r and c inputs are optional and need to be filled in if they are
% missing
if nargin < 4
    r = 0;
end
if nargin < 5
    c = 0;
end

if ~iscellstr(headers)
    error('Header must be cell array of strings')
end

 
if length(headers) ~= size(m,2)
    error('number of header entries must match the number of columns in the data')
end

%% write the header string to the file

%turn the headers into a single comma seperated string if it is a cell
%array, 
header_string = headers{1};
for i = 2:length(headers)
    header_string = [header_string,',',headers{i}];
end
%if the data has an offset shifting it right then blank commas must
%be inserted to match
if r>0
    for i=1:r
        header_string = [',',header_string];
    end
end

%write the string to a file
fid = fopen(filename,'w');
fprintf(fid,'%s\r\n',header_string);
fclose(fid);

%% write the append the data to the file

%
% Call dlmwrite with a comma as the delimiter
%
dlmwrite(filename, m,'-append','delimiter',',','roffset', r,'coffset',c);

end

