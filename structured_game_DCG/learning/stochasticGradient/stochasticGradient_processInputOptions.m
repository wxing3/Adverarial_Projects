function [n_sample, itr2chk, maxiteration, save_cycle, minibatch_size, n_worker, rate, decay, fTol, gTol, logPath, verboseI]=stochasticGradient_processInputOptions(o)

    o = toUpper(o);
    
    if isfield(o,'NSAMPLE')
        if ~isempty(getfield(o,'NSAMPLE')) %#ok<GFLD>
             n_sample = getfield(o,'NSAMPLE'); %#ok<GFLD>
        else
            error('Sample number must be given.'); 
        end
    else
       error('Sample number must be given.'); 
    end
    %% Control parameters
    % How many iteration to check for stability
    itr2chk = getOpt(o,'CHECK_AFTER',20); 

    % start training
    max_update = getOpt(o,'MAX_ITR',2000000);
    maxiteration = ceil(max_update / n_sample); % each pass through all inputs

    % save intermediate data periodically
    save_cycle =  getOpt(o,'SAVE_CYCLE',1000); % after this number of check, save the intermediate results

    % mini batch
    minibatch_size = getOpt(o,'MINIBATCH_SIZE',10);    % 10 samples each worker/process
    n_worker = getOpt(o,'WORKER_NUM',8);           % 8 processes/workers

    % adagrad prams
    rate = getOpt(o,'RATE',1);   % 1 maximum cases
    decay = getOpt(o,'DECAY',1);  % may or may not used
    
    % Stopping cretiria
    fTol = getOpt(o,'FTOL',10^-6);
    gTol = getOpt(o,'GTOL',10^-6);
    
    % Logging setting
    logPath = getOpt(o,'LOG_PATH','');
    verboseI = getOpt(o,'VERBOSE',true);
end

function [v] = getOpt(options,opt,default)
    if isfield(options,opt)
        if ~isempty(getfield(options,opt)) %#ok<GFLD>
            v = getfield(options,opt); %#ok<GFLD>
        else
            v = default;
        end
    else
        v = default;
    end
end

function [o] = toUpper(o)
    if ~isempty(o)
        fn = fieldnames(o);
        for i = 1:length(fn)
            o = setfield(o,upper(fn{i}),getfield(o,fn{i})); %#ok<GFLD,SFLD>
        end
    end
end