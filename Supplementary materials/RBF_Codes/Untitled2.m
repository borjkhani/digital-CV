load CellCycle  % matlab data structure containing the cell cycle data


[N,P] = size(AD);  % number of rows and columns
%N = 1000; % uncomment this to reduce the size of the data set
%AD = AD(1:N,:);

% help the user get some clusters
fprintf('This module clusters the cell cycle data by k-means.\n');
fprintf('There are %d observations in this data set\n', N);
nclust = input('How many clusters would you like: ');


% normalise the data
for i = 1:N, AD(i,:) = AD(i,:) - mean(AD(i,:)); end  % center the data
for i = 1:N, AD(i,:) = AD(i,:)/norm(AD(i,:)); end    % standardize the variance


% center contains the cluster centroids
% U is an indicator matrix mapping genes to clusters
[center, U, obj_fcn] = kmeans(AD, nclust);

% leave this in to plot the cluster centroids
figure
plot(1:14,center)
title('Cluster representations')

% here are some things that you might want to do

%show_clusters(AD,U)          % show plots of the clusters interactively
%show_image(AD,U,[1,2,3,4,5]) % show an image of the clusters
%show_genes(AD,U,3)  % return the rows of AD corresponding to cluster 3
%show_genes(Des,U,3) % return the Affymetrix identifiers of genes in cluster 3
