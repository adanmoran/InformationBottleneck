%% Main Clustering Search Algorithm
% If data sampled from a non-symmetric gaussian mixture exists, run that.
% Otherwise, generate the data and save it.
dataPointsFile = 'gmNonSymPoints.mat';
if exist(dataPointsFile, 'file')  
    load(dataPointsFile);
else
    % Generate 300 points from:
    % - a gaussian at (0,0) with covariance matrix = [0.5 0.1; 0.1 0.5] (i.e. 
    % its components are weakly positively correlated) and have low
    % variance)
    % - a gaussian at (0,-5) with covariance matrix = [5 0; 0 1] (i.e. its
    % components are independent, and points appear more along X)
    % - a gaussian at (3,3) with covariance matrix = [1 -0.9; -0.9 1] (i.e.
    % its components are highly negatively correlated)
    % 
    % All of these are equally likely to occur
    
    % Create the covariance matrices
    sigma1 = [0.5 0.1; 0.1 0.5];
    sigma2 = [5 0; 0 1];
    sigma3 = [1 -0.9; -0.9 1];
    sigma = cat(3,sigma1,sigma2,sigma3);
    % Create the matrix of means
    mu1 = [0 0];
    mu2 = [0 -5];
    mu3 = [3 3];
    mu = cat(1,mu1,mu2,mu3);
    % Sample 200 points
    [points, gm] = gmpoints(200, mu, sigma);
    % Now save the data for future use.
    gmNonSymPoints.points = points;
    gmNonSymPoints.gm = gm;
    save(dataPointsFile,'gaussianNonSymPoints');
end

% Choose which gammas we will use for our data
gammas = [0.1 0.5 1 1.5 2 5];

% Set up our array of structs
gmNonSymStruct = cell(size(gammas));

% Set the clustering parameters
s = 1;
M = 30;

% Loop over all the gammas which have not been completed.
i = 1;
for gamma = gammas
    % Remove all figures at each loop, since they get in the way and we're
    % just doing data right now.
    close all;
    % If we have not already done the struct
    if size(gmNonSymStruct{i},1) == 0
        fprintf('===================================\n');
        fprintf('Running clustering for gamma = %.2f\n',gamma);
        fprintf('===================================\n');
        fprintf('* Will save into location %d\n',i);
        % Run the clustering on this gamma with the chosen beta
        QS = bottlecluster(gmNonSymPoints.points,s,M,gamma);
        QS.gm = gmNonSymPoints.gm;
        % Set the output structure to contain this struct
        gmNonSymStruct{i} = QS;
        
        % Save the output structure's progress so far
        save('gmNonSymData.mat','gmNonSymStruct');
    end
    i = i + 1;
end