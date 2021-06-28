function [] = export_featmap(dataset)

data_specs = set_specs(dataset);

for s = 1:length(data_specs.subj)

    % Define some metadata.
    subjNo = data_specs.subj(s);
   
    % Load subject data file.
    subjFN = dir(strcat(data_specs.behav,strcat(strcat('Subj', num2str(subjNo)), '*.mat'))).name; 
    fp = fullfile(data_specs.behav, subjFN);
    load(fp);
    
    % Get number of runs.
    n_runs = length(Data);
    
    % Prepare feature map array.
    features_aoi_map = [];
   
    for r = 1:n_runs 
        
        % Grab data for this run.
        this_run_data = Data{r}; 
        this_run_stim = this_run_data.Stimuli;
        this_run_row_order = this_run_data.RowOrder(1,:);
        
        % Get number of trials.
        n_trials = length(this_run_data.StimOn);
       
        % Pre-allocate array for trialwise AOI mapping. 
        featmap = NaN(n_trials,9);

        % Iterate over trials.
        for t = 1:n_trials
        
            % Grab the task features and convert to 1-9 feature notation. 
            % 1-3 are Faces, 4-6 are Houses, 7-9 are Tools.
            stimuli_trial = squeeze(this_run_stim(t,:,:))';
            hlp = [zeros(1,3); 3*ones(1,3); 6*ones(1,3)];
            features_trial_aux = stimuli_trial + hlp; 

            % Use row order to remap dimensions to their correct position.
            features_trial = NaN(3,3);
            for f = 1:3
                features_trial(f,:) = (features_trial_aux(find(this_run_row_order == f),:));
            end

            % Map features to AOIs.
            % AOI map for FHT experiment: 
            % [1][4][7] 
            % [2][5][8]
            % [3][6][9]

            featmap(t,:) = [features_trial(:,1)' features_trial(:,2)' features_trial(:,3)']; 

        end
        
        % Stack this run's featmap.
        features_aoi_map = [features_aoi_map; featmap];
        
    end

    op = fullfile(data_specs.output,strcat(num2str(subjNo),'featmap.mat'));
    save(op,'features_aoi_map');
    
end

end
