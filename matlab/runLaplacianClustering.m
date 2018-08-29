%% Main Clustering Search Algorithm
% If data sampled from a non-symmetric gaussian mixture exists, run that.
% Otherwise, generate the data and save it.
dataPointsFile = 'lapPoints.mat';
if exist(dataPointsFile, 'file')  
    load(dataPointsFile);
else
    % Generate 100 points from each of the following, for a total of 200
    % points:
    % - A laplacian distribution with mu = [0 0] and sigma = I
    % - A laplacian distribution with mu = [4 0] and sigma = [2 0; 0 0.5]
    
    % Create the covariance vectors
    sigma1 = [1 1];
    sigma2 = [2 0.5];
    sigma = cat(1,sigma1,sigma2);
    % Create the matrix of means
    mu1 = [0 0];
    mu2 = [4 0];
    mu = cat(1,mu1,mu2);
    % Sample 200 points (100 from each one) and save both the points and
    % the laplacian mixture
    [points, lm] = lappoints(200, mu, sigma);
    % Now save the data for future use.
    lapPoints.points = points;
    lapPoints.lm = lm;
    save(dataPointsFile,'lapPoints');
end

% Choose which gammas we will use for our data
gammas = [0.1 0.5 1 1.5 2 5];

% Set up our array of structs
if exist('lapData.mat','file')
    load('lapData.mat');
else
    lapStruct = cell(size(gammas));
end

% Set the clustering parameters
s = 1;
M = 30;

% Loop over all the gammas which have not been completed.
i = 1;
% Display the points and the true distribution
figure;
scatter(lapPoints.points(:,1), lapPoints.points(:,2));
hold on
fcontour(@(x,y) lapPoints.lm([x y]), [-3 6]);
hold off
title('True Contour Diagram: Laplacian Data');
xlabel('x');
ylabel('y');

% Display the true pdf
grid = cell(2,1);
[grid{:}] = ndgrid(-3:0.01:8,-3:0.01:3);
components = cellfun(@(axis) reshape(axis,[numel(axis) 1]), grid, ...
'UniformOutput', false);
gridPoints = cat(2,components{:});
z = reshape(lapPoints.lm(gridPoints), size(grid{1}));
% Show the mesh
figure;
mesh(grid{:},z);
title('True PDF: Laplacian Data');
xlabel('X');
ylabel('Y');
zlabel('Z');

for gamma = gammas
    % Remove all figures at each loop, since they get in the way and we're
    % just doing data right now.
    close all;
    % If we have not already done the struct
    if size(lapStruct{i},1) == 0
        fprintf('===================================\n');
        fprintf('Running clustering for gamma = %.2f\n',gamma);
        fprintf('===================================\n');
        fprintf('* Will save into location %d\n',i);
        % Run the clustering on this gamma with the chosen beta
        QS = bottlecluster(lapPoints.points,s,M,gamma,2);
        QS.lm = lapPoints.lm;
        % Set the output structure to contain this struct
        lapStruct{i} = QS;
        
        % Save the output structure's progress so far
        save('lapData.mat','lapStruct');
    end
    i = i + 1;
end