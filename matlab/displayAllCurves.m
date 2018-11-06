%% Generate IB and DIB on RDIB Plane
% This script computes some values for the IB and DIB, then displays them
% on the RDIB plane for a given gamma. This is used for the report.

% Set the inputs to the bottlecurve
Pxy = [0.2 0.1; 0.5 0.05; 0.05 0.1];
N = 10;
delta = 1/40;
epsilon = 10^-8;
display = 'none';
Bs = [0 0.25 0.5 0.75 1 2 3 4 5 6 7 8 10 ...
      10.5 11 12 13 14 15 16 17 18 19 20 ...
      30 40 50 100 Inf];
gamma = 0.8;

% Get gamma as a string, replace decimal places with underscores
strGamma = replace(num2str(gamma),'.','_');
% Folder where we will save images
folder = '../../images/rdib/';
% Flag to choose whether these files are saved.
save = true;

% Compute the IB curve
[~,IB.Ht,IB.Ixt,IB.Iyt,IB.Bs,IB.Qt] = bottlecurve(...
    Pxy,N,1,1,delta,epsilon,display,Bs);

% Compute the DIB curve
[~,DIB.Ht,DIB.Ixt,DIB.Iyt,DIB.Bs,DIB.Qt] = bottlecurve(...
    Pxy,N,0,1,delta,epsilon,display,Bs);

% Compute the RDIB curve
[RDIB.Hgt,RDIB.Ht,RDIB.Ixt,RDIB.Iyt,RDIB.Bs,RDIB.Qt] = bottlecurve(...
    Pxy,N,0,gamma,delta,epsilon,display,Bs);

% Create vectors for H_gamma(T) for both the IB and DIB, to be able to
% display them all on the same plane
IB.Hgt = zeros(size(IB.Ixt));
DIB.Hgt = zeros(size(DIB.Ht));
numDistributions = length(Bs);

% Compute these H_gamma(T) values
for distribution = 1:numDistributions
    IB.Hgt(distribution) = entropy(IB.Qt(:,distribution), gamma);
    DIB.Hgt(distribution) = entropy(DIB.Qt(:,distribution), gamma);
end

% Get the distribution of X so we can compute upper limits on
% horizontal axes.
Px = makeDistribution(sum(Pxy,2));

% Get the upper limits for the ib and dib plane, given by H(X), as well
% as the upper limit for the gib plane, given by H_gamma(X).
Hx = entropy(Px);
Hgx = entropy(Px,gamma);

% Compute the mutual information between X and Y, which is the upper
% limit of the vertical axis on all planes.
Pygx = makeDistribution(Pxy ./ Px, 2);
Ixy = mi(Pygx,Px);
    
% Plot everything on the IB plane
ibf = figure;
hold on
plot(IB.Ixt, IB.Iyt, 'blue', ...
     DIB.Ixt, DIB.Iyt, 'red', ...
     RDIB.Ixt, RDIB.Iyt, 'magenta');
% Put in the titles, legend, and axes labels
xlabel('I(X;T)');
ylabel('I(T;Y)');
title(sprintf('IB Plane for |X|=%d and |Y|=%d',...
    size(Pxy,1),size(Pxy,2)));
% Plot the box within which the curve should lie, which is given by
% H(X) and I(X;Y)
plot([0,Hx],[Ixy,Ixy], 'black'); % I(X;Y) horizontal line
plot([Hx,Hx],[0,Ixy], 'green'); % H(X) vertical line
hold off;
% Plot the legend in the bottom right corner with no background or
% outline
legend({'IB','DIB','RDIB','I(X;Y)','H(X)'},'Location','Southeast');
legend('boxoff');

% Plot everything on the DIB plane
dibf = figure;
hold on
plot(IB.Ht, IB.Iyt, 'blue',...
     DIB.Ht, DIB.Iyt, 'red',...
     RDIB.Ht, RDIB.Iyt, 'magenta');
% Put in the titles, legend, and axes labels
xlabel('H(T)');
ylabel('I(T;Y)');
title(sprintf('DIB Plane for |X|=%d and |Y|=%d',...
    size(Pxy,1),size(Pxy,2)));
% Plot the box within which the curve should lie, which is given by
% H(X) and I(X;Y)
plot([0,Hx],[Ixy,Ixy], 'black'); % I(X;Y) horizontal line
plot([Hx,Hx],[0,Ixy], 'green'); % H(X) vertical line
hold off;
% Plot the legend in the bottom right corner with no background or
% outline
legend({'IB','DIB','RDIB','I(X;Y)','H(X)'},'Location','Southeast');
legend('boxoff');

% Plot everything on the RDIB plane
rdibf = figure;
hold on
plot(IB.Hgt, IB.Iyt, 'blue',...
     DIB.Hgt, DIB.Iyt, 'red',...
     RDIB.Hgt, RDIB.Iyt, 'magenta');
% Put in the titles, legend, and axes labels
xlabel('H(T)');
ylabel('I(T;Y)');
title(sprintf('Renyi DIB Plane for |X|=%d and |Y|=%d',...
    size(Pxy,1),size(Pxy,2)));
% Plot the box within which the curve should lie, which is given by
% H(X) and I(X;Y)
plot([0,Hgx],[Ixy,Ixy], 'black'); % I(X;Y) horizontal line
plot([Hgx,Hgx],[0,Ixy], 'green'); % H(X) vertical line
hold off;
% Plot the legend in the bottom right corner with no background or
% outline
legend({'IB','DIB','RDIB','I(X;Y)',sprintf('H_{%.2f}(X)',gamma)},'Location','Southeast');
legend('boxoff');

% Save the files
ibfile = 'ib_plane_gamma_';
ibfile = strcat(ibfile, strGamma);
ibpathfig = strcat(folder, ibfile, '.fig');
ibpathpng = strcat(folder, ibfile, '.png');
saveas(ibf, ibpathfig);
saveas(ibf, ibpathpng);

dibfile = 'dib_plane_gamma_';
dibfile = strcat(dibfile, strGamma);
dibpathfig = strcat(folder, dibfile, '.fig');
dibpathpng = strcat(folder, dibfile, '.png');
saveas(dibf, dibpathfig);
saveas(dibf, dibpathpng);

rdibfile = 'rdib_plane_gamma_';
rdibfile = strcat(rdibfile, strGamma);
rdibpathfig = strcat(folder, rdibfile, '.fig');
rdibpathpng = strcat(folder, rdibfile, '.png');
saveas(rdibf, rdibpathfig);
saveas(rdibf, rdibpathpng);