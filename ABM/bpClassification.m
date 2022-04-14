%clc;
%clear;
restoredefaultpath;
addpath(genpath(pwd));
%%Initialization **************************************************************************************
global n_frames;
global n_features;
global n_nodes;
global n_edges;
load('C:\Users\Sima\Documents\adversarialBipartitMatching\data-matching\data\Pedcross2-Bahnhof.mat');
n_features=size(X_train,1);
n_frames=size(X_train,4);
n_nodes=size(X_train,2);
folder=' ';
% profile -memory on;
% Please set the matrices in a way that the last dimention defines the
% number of entities in the matrix%
%dataAddress= 'C:\Users\Sima\Documents\adversarialBipartitMatching\dataSet\2DMOT2015\train\ETH-Bahnhof\gt\gt.txt';
%[frames,frames_randomized,features_edges,groundTruth,n_nodes,n_frames]=extractData(dataAddress);
n_edges=n_nodes^2;
save_after=20;
maxiteration=10000;
batchSize=25;
%Here, n_nodes is the max number of matching nodes n consequtive frames.
global weight_size_edge;
weight_size_edge=n_features;

theta=abs(randn(weight_size_edge,1));
thetea_all=zeros(weight_size_edge,maxiteration);

avg_objective_value_maximizer=zeros(maxiteration,1);
avg_grads_magnitude_pairwise = zeros(maxiteration,1); %

sum_game_value_maximizer_batch=0;
sum_objective_value_maximizer_batch=0;

sum_objective_value_maximizer_total=0; % the sum of objective function values over training examples
sum_game_value_maximizer_total=0;

sum_grad_batch_edges = zeros (weight_size_edge,1);  % % the sum of gradients over training examples in each batch
avg_grad_batch_edges= zeros (weight_size_edge,1);

% adagrad ********************************************
%https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
autocorr = 0.8;
fudge_factor=1e-2; %for numerical stability
master_stepsize = 0.1;
historical_grad= zeros (weight_size_edge,1);
%********************************************
%% Training
for itr = 1:maxiteration
    itr
    sum_objective_value_maximizer_total=0;
    sum_game_value_maximizer_total=0;
    order = randperm ( n_frames);
    for idx=1:batchSize:n_frames
        sum_objective_value_maximizer_batch=0;
        sum_game_value_maximizer_batch=0;
        for bindex=idx:(idx+batchSize-1)
            if bindex > n_frames
                bindex=idx;
            end
            ind=order(bindex);
            [sample_grad_edges,game_value_maximizer,objective_value_maximizer]=game_step_bp(X_train(:,:,:,ind),Y_train(:,:,ind),theta);
            
            sum_grad_batch_edges=sum_grad_batch_edges+sample_grad_edges(:,1,1);
          %  reshape(sample_grad_edges(1,1,:),2,1)
            sum_game_value_maximizer_batch=sum_game_value_maximizer_batch+game_value_maximizer;
            sum_objective_value_maximizer_batch=sum_objective_value_maximizer_batch+objective_value_maximizer;
        end
        
        
        avg_grad_batch_edges=sum_grad_batch_edges./batchSize;
        sum_grad_batch_edges= zeros(weight_size_edge,1);
        
        sum_game_value_maximizer_total=sum_game_value_maximizer_total+(sum_game_value_maximizer_batch./batchSize);
        sum_objective_value_maximizer_total=sum_objective_value_maximizer_total+(sum_objective_value_maximizer_batch./batchSize);
        %% adagrad
        
        historical_grad=historical_grad+(avg_grad_batch_edges.^2);
        adjusted_grad_edges=reshape(sample_grad_edges(:,1,1), n_features,1)./(sqrt(historical_grad)+fudge_factor);
        
        %% gradient update
        theta=theta- master_stepsize * adjusted_grad_edges;
       % theta=max(theta,0);

    end
        %% Recording
        thetea_all(:,itr)=theta;
         avg_game_value_maximizer(itr) = sum_game_value_maximizer_total/batchSize;  % average
    sum_game_value_maximizer_total=0;
    
    avg_objective_value_maximizer(itr)= sum_objective_value_maximizer_total/batchSize; 
      
        if (objective_value_maximizer<0)
            itr
        end
            sum_objective_value_maximizer_total=0;
        grads_magnitude_edges(itr) = (1/weight_size_edge)*sum(sum(sum(abs(sample_grad_edges))));
        
        %%
         
        if (itr == maxiteration)
            
            disp('exceeded maximum iteration');
            
            break_condition = 'exceeded maximum iteration';
            
        end
        if( mod (itr, save_after) == 0 ) % based on data size, itr takes variable times. so save on update count instead
            
            %*****************************************************************
            
            fig=figure('Visible','off','Position', [0 0 1024 800]);
            
            plot(avg_objective_value_maximizer(1:itr));
            
            figName=strcat(folder,'Objectiveplot.png');
            
            saveas(fig, figName);
            %*****************************************************************
            
            fig=figure('Visible','off','Position', [0 0 1024 800]);
            
            plot(grads_magnitude_edges(1:itr));
            figName=strcat(folder,'gradplot_edges.png');
            saveas(fig, figName);
            %*****************************************************************
            fig=figure('Visible','off','Position', [0 0 1024 800]);
            
            plot(thetea_all(:,1:itr)');
            figName=strcat(folder,'thetaplot_edges.png');
            
            %         figName='thetaplot_pairwise.png';
            
            saveas(fig, figName);
            %*****************************************************************
            
            fig=figure('Visible','off','Position', [0 0 1024 800]);
            
            plot( avg_game_value_maximizer(1:itr));
            
            %         figName='game_values.png';
            figName=strcat(folder,'game_values.png');
            
            saveas(fig, figName);
            %*****************************************************************
            
            save theta_edges theta
        end
    end
    %% logging
    
    fig=figure('Visible','off','Position', [0 0 1024 800]);
    
    plot(grads_magnitude_features(1:itr));
    
    saveas(fig,'finalgradplot_gpu.png');
    
    save('lastrunallFeatures_gpu.mat');
    
    fileID = fopen('output_gpu.txt','w');
    
    fprintf(fileID, [break_condition '\n']);
    
