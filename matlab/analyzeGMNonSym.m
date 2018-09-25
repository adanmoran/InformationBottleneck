%% Analyze Data from Renyi's Clustering
% Load the data which was generated by the non-symmetric GM clustering. The
% contents should be:
% - a gaussian at (0,0) with covariance matrix = [0.5 0.1; 0.1 0.5]
% - a gaussian at (0,-5) with covariance matrix = [5 0; 0 1]
% - a gaussian at (3,3) with covariance matrix = [1 -0.9; -0.9 1]
load('gmNonSymData.mat');

% Folder where we will save images
folder = '../../images/clustering/nonsym_gaussian/same_beta/';

% Set the distribution info (gm, points), which is the same for all of them
gm = gmNonSymStruct{1}.gm;
points = gmNonSymStruct{1}.points;

% If this is false, we use the kink angle already specified in the
% gmNonSymStruct object when plotting graphs. Otherwise, we use the selected
% input beta for ALL graphs (and recompute q(c|i), q(c|x), etc. without
% overwriting the old ones)
useInputBeta = true;
inputBeta = 2.0;
% If this is true, we will allow the user to redo the kink beta. Note that
% useInputBeta will take priority over this (i.e. if useInputBeta is true,
% we will run the loop which plots the IB plane but will not redo the kink
% beta).
redoKinkBeta = false;
% If this is true, we show the partition lines for q(c|x) (filled in
% colour).
showPartition = false;

% Loop through each Renyi-DIB and plot the results of the clustering for
% the kinked beta
for i = 1:size(gmNonSymStruct,2)
    % Set the structure for this gamma
    resultStruct = gmNonSymStruct{i};
    % Get the grid on which we have computed the points X
    grid = resultStruct.grid;
    
    % Get the X data, which is the points that partition the grid
    X1 = reshape(grid{1},size(resultStruct.fx));
    X2 = reshape(grid{2},size(resultStruct.fx));
    X = cat(2,X1,X2);
    
    % Get gamma as a string, replace decimal places with underscores
    strGamma = replace(num2str(resultStruct.gamma),'.','_');
    
    % Find out if this structure does not have certain elements. If
    % not, display the information plane and show each beta for it to
    % choose the desired kink angle, q(c|i), q(c|x), q(c), etc...
    if ~isfield(resultStruct, 'kinkBeta') || ...
      ~isfield(resultStruct, 'Qcgx') || ...
      ~isfield(resultStruct, 'Qcgi') || ...
      ~isfield(resultStruct, 'Qc') || ...
      useInputBeta || ...
      redoKinkBeta
        % Get all the unique components of the curve, as there are often
        % doubles
        [uniqueHga, uniqueHgaIndex] = unique(resultStruct.Hga);
        uniqueIyt = resultStruct.Iyt(uniqueHgaIndex);
        
        % Plot the information plane
        plane = figure;
        plot(uniqueHga, uniqueIyt);
        xlabel(sprintf('H_{%.2f}(T)',resultStruct.gamma));
        ylabel('I(T;Y)');
        title('Renyi-DIB Information Plane (Unique Hgas Only)');
        % Save this figure as a PNG
        planepng = strcat(folder,'plane_',strGamma,'.png');
        saveas(plane, planepng);
        
        % Show the beta values at each point
        betaStrings = string(resultStruct.betas(uniqueHgaIndex));
        text(uniqueHga,uniqueIyt,betaStrings);
        
        % Get the kink angle as an input. If we chose to use an input beta
        % for all of the graphs instead, just assign that to the kink angle
        % as it won't be overwritten.
        if useInputBeta
            fprintf('Setting Beta to an input.');
            kink = inputBeta;
        else
            kink = input('What Beta is the kink angle? ');
        end
        
        % Save the kink angle in the structure
        resultStruct.kinkBeta = kink;
        
        % Compute the cluster for this kink beta
        fprintf('Finding q(c|i) and q(c) for beta = %.2f...\n',kink);
        [Qcgi, Qc] = optimalbottle(resultStruct.Pix, resultStruct.gamma,...
                                   0, kink);
    
        % Remove the clusters that have zero probability
        Qcgi = Qcgi(:,Qc ~= 0);
        Qc = Qc(Qc ~= 0);
    
        % Now multiply the matrix by the c-values to get the cluster locations
        % for each i
        clusterLabels = transpose(1:size(Qcgi,2));
        c = Qcgi * clusterLabels;
        
        % Update the struct with q(c|i), q(c), and cluster labels c
        resultStruct.Qcgi = Qcgi;
        resultStruct.Qc = Qc;
        resultStruct.c = c;
        
        % Find the distribution q(c|x) = sum_i P(i,x)q(c|i) / P(x)
        QcgxPx = transpose(resultStruct.Pix) * Qcgi;
        Qcgx = makeDistribution(QcgxPx ./ resultStruct.fx, 2);  
        
        % Update the struct with q(c|x)
        resultStruct.Qcgx = Qcgx;
        
        % Make note that we changed the gmNonSymStruct if we are going to save
        % it (in the kink angle case)
        if ~useInputBeta
            gmNonSymStruct{i} = resultStruct;
            changed = true;
        end
        
        % Delete the figure and continue
        close(plane);
    end
    
    % Get q(c|x) from the struct data
    Qcgx = resultStruct.Qcgx;
    
    % This is a probability distribution, so let's take the argmax
    % along each row to get the clusters for the grid, as per the DIB
    [~,clusterXDIB] = max(Qcgx,[],2);
    % Reshape this to a 2D grid
    clusterXDIB = reshape(clusterXDIB,size(grid{1}));
    
    % Now cluster the X based on the GM distribution
    clusterXGM = gm.cluster(X);
    clusterXGM = reshape(clusterXGM, size(grid{1}));
    
    % Display the clustering at the kink beta
    fig = figure;
    hold on
    if showPartition
        % Show the contour plot of the DIB distribution, where we only show
        % the levels associated with the clusters (to avoid interpolation
        % and extra lines). Colour each region in by using contourf instead
        % of contour.
        lowestDIB = min(min(clusterXDIB));
        highestDIB = max(max(clusterXDIB));
        contourf(grid{1},grid{2},clusterXDIB,lowestDIB:highestDIB);
    end
    
    % Now plot the contour of the GM just as a contour line, without the
    % fill. Make it a black dashed line to see it easily.
    lowestGM = min(min(clusterXGM));
    highestGM = max(max(clusterXGM));
    contour(grid{1},grid{2},clusterXGM, lowestGM:highestGM, 'k--');
    
    % Show the points themselves
    gscatter(points(:,1),points(:,2),resultStruct.c)
    
    % Set the title and labels
    title(sprintf('DIB vs GM Partition, gamma = %.2f, kinkBeta = %.2f',...
           resultStruct.gamma, resultStruct.kinkBeta));
    xlabel('X-Coordinate');
    ylabel('Y-Coordinate'); 
    
    if showPartition
        legend('DIB Partition', 'GM Partition', 'Clusters')
    else
        legend('GM Partition', 'Clusters');
    end
    hold off;
    
    % Compute the filename where we are saving the image
    file = 'gm_ns_cluster_';
    file = strcat(file, strGamma);
 
    % Get the full filepath as a MATLAB figure and as a PNG
    filepathfig = strcat(folder,file,'.fig');
    filepathpng = strcat(folder,file,'.png');
    % Save as a MATLAB figure
    savefig(fig, filepathfig);
    % Save as a PNG
    saveas(fig, filepathpng);
    
    if i == 1
        % If this is the first loop, also plot the true pdf of the mixture,
        % and the true contour plot inluding points. These will not be
        % saved automatically

        % Display the true pdf
        trueFig = figure;
        fsurf(@(x,y) gm.pdf([x y]), ...
            [min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))])
        xlabel('X');
        ylabel('Y');
        zlabel('PDF');
        title('True PDF of Non-Symmetric GM at (0,0) + (0,-5) + (-3,-3)');
        
        % Display the true contour plot and the points clustered, in
        % colour.
        trueContour = figure;
        hold on;
        % Colour in the contour
        contourf(grid{1},grid{2},clusterXGM, lowestGM:highestGM);
        % Show the contour lines of the pdf
        fcontour(@(x,y) gm.pdf([x y]),...
            [min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))]);
        % Show the points
        gscatter(points(:,1),points(:,2),gm.cluster(points))
        hold off;
        xlabel('X');
        ylabel('Y');
        str = sprintf('True partition of %d points from Non-Symmetric GM at (0,0) + (0,-5) + (-3,-3)',...
            size(points,1));
        title(str);
        
        % Show the estimated pdf
        estimatedFig = figure;
        z = reshape(resultStruct.fx, size(grid{1}));
        surf(grid{:}, z);
        xlabel('X');
        ylabel('Y');
        zlabel('f(x,y)');
        title('GM Distribution of gaussians centered at each of the 300 x_i');
    end
    
end

% Save the struct, if it was changed
if changed
    save('gmNonSymData.mat','gmNonSymStruct');
end