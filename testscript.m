A = importdata('cleandata_students.mat'); 
B = importdata('noisydata_students.mat');


topology = [30 30];
lr = 0.01;
inc = 1.15;
dec = 0.46;

% Train and save
[BestPerf,BestErr,ErrNet,PerfNet,ErrInd,PerfInd] = TrainNetwork(A.x,A.y,  topology,lr,0,0,0,inc,dec);

save net.mat PerfNet


% Estimate Performance
[BestPerfC,ClassificationErrorC,PerfNetC,confMatrixC,recallC,precisionC,F1C] = EvaluateNetwork(A.x,A.y,  topology,lr,0,0,0,inc,dec);

[BestPerfN,ClassificationErrorN,PerfNetN,confMatrixN,recallN,precisionN,F1N] = EvaluateNetwork(B.x,B.y,  topology,lr,0,0,0,inc,dec);


% Test on entire data set
cleanPredictions = testANN(PerfNetC, A.x);
cleanError = size(find(cleanPredictions - A.y),1)/size(A.x,1); 

noisyPredictions = testANN(PerfNetN, B.x);
noisyError = size(find(noisyPredictions - B.y),1)/size(B.x,1); 


