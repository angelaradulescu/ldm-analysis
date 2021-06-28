function[data_specs] = set_specs(dataset)

switch dataset
                    
    case 1 % first LDM experiment
        
        data_specs.behav = '/Users/angelaradulescu/Dropbox/NYU/Research/LDM/RawData/';
        data_specs.ET = '/Users/angelaradulescu/Dropbox/NYU/Research/LDM/RawData/';
        data_specs.output = '/Users/angelaradulescu/Dropbox/NYU/Research/LDM/RawData/';
        
        data_specs.subj = [23, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 53, 54, 55, 57, 58, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71];
        
        data_specs.nRuns = 10;
        
end

end
