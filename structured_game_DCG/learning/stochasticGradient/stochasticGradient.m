function [bestX,bestF] = stochasticGradient(funObj,x0,options,varargin)
% Stochastic gradient used to find minimal value of a function.

if nargin < 3
    options = [];
end

x = x0;
n_feature = length(x0);

% Get Parameters
[ n_sample, itr2chk, maxiteration, save_cycle, minibatch_size, n_worker, rate, decay, fTol, gTol, logPath, verboseI] = ...
    stochasticGradient_processInputOptions(options);

%% setup the parpool (parallel pool)
% cluster = parcluster();
% t = tempname();
% mkdir(t);
% cluster.JobStorageLocation = t;
% if ~isempty(gcp('nocreate')) % >= 2013b
%     delete(gcp)
%     parpool(cluster,'AttachedFiles',[func2str(funObj) '.m']);
% else
%     parpool(cluster, cluster.NumWorkers,'AttachedFiles',{[func2str(funObj) '.m']});
% end

% adagrad
square_g = zeros( n_feature, 1 ) + 1e-20; % very small number, so that deviding does not give NaN

F = zeros(maxiteration,1); % prealocate max iteration
A = zeros(maxiteration,1); % prealocate max iteration
G = zeros(maxiteration,1); % prealocate max iteration
X = zeros(n_feature, maxiteration);

start_time = tic;
t = 0; % t is the number of the updating of x

% mini-batch
bestF = inf;
bestX = x0;

%% print
if verboseI
    fprintf('%10s %10s %15s %15s %15s\n','Iteration','FunEvals','Avg Step Length','Function Val','Gradient Norm');
end

for itr = 1:maxiteration
    
    sum_f = 0;          % the sum of objective function
    sum_a = 0;          % additional output
    sum_g = zeros(n_feature,1);  % sum of the gradient, to produce a average gradient
    update_count = 0;            % how many updates performed. variable based on sample per parloop. so needed.
    valid_count = 0;           % samples succeeded in providing valid information (no computation errors).
    
    % randomize indices
    order = randperm ( n_sample );
    
    for k = 1: minibatch_size*n_worker :n_sample % increase by batch size
        grad_count = zeros (n_worker, 1); % how many gradient computed. different for final batch.
        batch_g = zeros (n_feature, n_worker ); % the sum of gradients per worker
        batch_f = zeros (n_worker, 1);   % sum of objective-function per worker
        batch_a = zeros (n_worker, 1); % sum of additional output of objective-function per worker
        
        batch_size = minibatch_size * n_worker;
        if k + batch_size > n_sample % final batch is smaller
            batch_size = n_sample - k + 1;
        end
        batch_idx = order(k:k+batch_size-1); % separate, so that does not access the big matrix
        
        %parfor worker = 1:n_worker % parallel workers
        for worker = 1:n_worker % parallel workers
            % inside worker
            gc = 0;     % count the gradients
            b_f = 0;   % add the game_values
            b_a = 0;
            b_grad = zeros (n_feature, 1);  % add the gradients
            
            for idx = 1 + (worker-1)*minibatch_size : worker * minibatch_size
                if idx > batch_size
                    break; % final batch, out of index
                end
                
                
                % get function value and gradient
                
                sampleIdx = batch_idx(idx);
                
                [f, g, a, errorFlag] = funObj(sampleIdx,x,varargin{:});
               
                if ~errorFlag
                    b_f = b_f + f;
                    b_a = b_a + a;
                    b_grad = b_grad + g;
                    gc = gc + 1;
                    valid_count = valid_count + 1;
                end
            end
            
            
            grad_count(worker) = gc;
            batch_g(:, worker) = b_grad;
            batch_a(worker) = b_a;
            batch_f(worker) = b_f;
        end
        
        if sum(grad_count) == 0
            warning('No valid update in this batch.');
            continue;
        end

        % after parallel loop
        avg_g = sum (batch_g, 2) / sum (grad_count);
        
        % game_val
        sum_f = sum_f + sum(batch_f);
        
        %addtional_output
        sum_a = sum_a + sum(batch_a);
        
        % adagrad
        sum_g = sum_g + avg_g;   % u_t
        square_g = square_g + avg_g .^ 2; % G_t
        
        step_size =  decay * rate ./ sqrt(square_g);
        
        x = x - step_size .* avg_g;
        
        update_count = update_count + 1; % update count in one iteration to compute avg gradient per pass through
        
        t = t + 1; % overall update count
        
        decay = 1000/(1000+t);
    end
    
    X(:,itr) = x;
    A(itr) = sum_a / valid_count;
    F(itr) = sum_f / valid_count;   % keep track
    
    % Find minimal value
    if F(itr) < bestF
        bestF = F(itr);
        bestX = x;
    end
    
    
    % stopping criteria
    if ( itr > itr2chk )
        
        itr1 = itr2chk/2;
        
        v1 = sum( F(itr-itr2chk+1:itr-itr1) )/ itr1;
        v2 = sum( F(itr-itr1+1 : itr) ) / itr1;
        
        if abs ( v1 - v2 ) <= fTol % check 2 portion
            disp('Function value breaks');
            break_condition = 'Function value breaks';
            break;
        end
    end
    
    % stopping criteria
    sum_g = sum_g  / update_count; % average
    G(itr) = norm(sum_g,1);   % keep track, grads are column vectors
    if ( itr > itr2chk ) % if enough results
        if sum_g <= gTol
            disp('Gradient breaks');
            break_condition = 'Gradient breaks';
            break;
        end
    end
    
    %% print
    if verboseI
       fprintf('%10d %10d %15.5e %15.5e %15.5e\n',itr,t,mean(step_size),F(itr),G(itr));
    end
    
    %% debugging
    if (itr == maxiteration)
        disp('Exceeded maximum iteration')
        break_condition = 'Exceeded maximum iteration';
    end
    
    if( mod (t, save_cycle) == 0 ) % based on data size, itr takes variable times. so save on update count instead
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        plot(F(1:itr));
        saveas(fig, [logPath 'fplot.png']);
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        plot(G(1:itr));
        saveas(fig, [logPath 'gplot.png']);
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        plot(A(1:itr));
        saveas(fig, [logPath 'aplot.png']);
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        plot(X(:,1:itr)');
        saveas(fig, [logPath, 'xplot.png']);
        
        save([logPath 'F.mat'], 'X', '-v7.3');
        save([logPath 'G.mat'], 'X', '-v7.3');
        save([logPath 'A.mat'], 'X', '-v7.3');
        save([logPath 'X.mat'], 'X', '-v7.3');
        save([logPath 'lastrun.mat'], '-v7.3');
    end
    
end

% Save final results
fig=figure('Visible','off','Position', [0 0 1024 800]);
plot(F(1:itr));
saveas(fig, [logPath 'fplot.png']);

fig=figure('Visible','off','Position', [0 0 1024 800]);
plot(G(1:itr));
saveas(fig, [logPath 'gplot.png']);

fig=figure('Visible','off','Position', [0 0 1024 800]);
plot(A(1:itr));
saveas(fig, [logPath 'aplot.png']);

fig=figure('Visible','off','Position', [0 0 1024 800]);
plot(X(:,1:itr)');
saveas(fig, [logPath, 'xplot.png']);

save([logPath 'F.mat'], 'X', '-v7.3');
save([logPath 'G.mat'], 'X', '-v7.3');
save([logPath 'A.mat'], 'X', '-v7.3');
save([logPath 'X.mat'], 'X', '-v7.3');
save([logPath 'lastrun.mat'], '-v7.3');

% log time
fileID = fopen([logPath 'output.txt'],'w');
fprintf(fileID, [break_condition '\n']);
fprintf(fileID, 'minimal_val = %f \n', F(itr));
fprintf(fileID, 'time = %f sec \n', toc (start_time));
fclose(fileID);