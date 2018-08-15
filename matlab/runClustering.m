%% Main Clustering Search Algorithm
% Load our data, which consists of 100 points sampled over two
% gaussians at [0,0] and [3,0] with Sigma = I.
load('highPoints.mat');

% Choose which gammas we will use for our data
gammas = [0.01 0.1 0.25 0.5 1 2 5 10 100];

% Set up our array of structs
highpStruct = cell(size(gammas));

% Fill in which gammas have already been done.
load('highpg01.mat');
[~,loc01] = max(gammas == 0.01);
highpStruct{loc01} = highpg01;
load('highpg05.mat');
[~,loc05] = max(gammas == 0.5);
highpStruct{loc05} = highpg05;
load('highpg1.mat');
[~,loc1] = max(gammas == 1);
highpStruct{loc1} = highpdib;
load('highpg2.mat');
[~,loc2] = max(gammas == 2);
highpStruct{loc2} = highpg2;

% Set the clustering parameters
s = 1;
M = 25;

% Choose which beta to use for the initial cluster observation. This should
% ideally be whichever beta was optimal for gamma = 1.
beta = 1.5;

% Loop over all the gammas which have not been completed.
i = 1;
for gamma = gammas
    % Remove all figures at each loop, since they get in the way and we're
    % just doing data right now.
    close all;
    % If we have not already done the struct
    if size(highpStruct{i},1) == 0
        fprintf('===================================\n');
        fprintf('Running clustering for gamma = %.2f\n',gamma);
        fprintf('===================================\n');
        fprintf('* Will save into location %d\n',i);
        % Run the clustering on this gamma with the chosen beta
        QS = bottlecluster(highPoints.points,s,M,gamma,beta);
        QS.gm = highPoints.gm;
        % Set the output structure to contain this struct
        highpStruct{i} = QS;
    end
    i = i + 1;
end

% Save the output structure
save('highpData.mat','highpStruct');