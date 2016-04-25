
clear;

fileId = 2;

player1 = displayResult(fileId, 4, 'A', 2048, 4096, 2, 32,0,'kmeans');
player2 = displayResult(fileId, 4, 'A', 2048, 4096, 2, 32,0,'gmm');
% % 
player1 = displayResult(fileId, 4, 'A', 2048, 4096, 2, 32,1,'kmeans');
player2 = displayResult(fileId, 4, 'A', 2048, 4096, 2, 32,1,'gmm');