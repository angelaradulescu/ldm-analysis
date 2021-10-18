
function [] = export_events(dataset)
% Compatible with NivLink 0.2

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
    
    % Initialize arrays.
    ids = [];
    onsets = [];
    offsets = [];
    durations = [];

    % Loop through runs. 
    for r = 1:n_runs

       % Grab data for this run.
       this_run_data = Data{r}; 

       % Get number of trials.
       n_trials = length(this_run_data.StimOn);
       
       % Preallocate run onset array.
       run_ids = NaN(n_trials,1);
       run_onsets = NaN(n_trials,1);
       run_offsets = NaN(n_trials,1);
       run_durations = NaN(n_trials,1);

       % Loop through trials.
       for t = 1:n_trials 
            run_ids(t) = r;
            % Align trial onsets relative to first onset in the block (aligned to Start messages in raw eyetracking).
            run_onsets(t) = this_run_data.StimOn(t) - this_run_data.StimOn(1);
            % Compute actual duration.
            trial_duration = this_run_data.OutcomeOn(t) + 2 - this_run_data.StimOn(t); 
            % Display numerical error. 
            err = trial_duration - 4;
            warning_text = strcat(sprintf('off by %0.2f',err*1000),'ms');
            disp(warning_text);
            % Record duration. 
            run_durations(t) = trial_duration;
            % Record trial offsets.
            run_offsets(t) = run_onsets(t) + trial_duration;
       end

       ids = [ids; run_ids];
       onsets = [onsets; run_onsets];
       offsets = [offsets; run_offsets];
       durations = [durations; run_durations];    

    end

    % We don't need the duration. 
    events_array = [ids, onsets, offsets];

    op = fullfile(data_specs.ET,strcat(num2str(subjNo),'events.mat'));
    save(op, 'events_array');
    
end

end