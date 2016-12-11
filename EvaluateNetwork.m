function [BestPerf,BestErr,PerfNet,confusionMatrix, recall, precision, F1] = EvaluateNetwork(data,y,  topology,lr,lrinc,lrdec,mc,inc,dec)

data = transpose(data);
[labels] = mapToTargets(y);
labels = transpose(labels);
Indices = importdata('noisy_indices.mat');
BestPerf = 1e10;
BestErr = size(data,1);
confSum = zeros(6,6); 
recall = zeros(6,1);
precision = zeros(6,1); 
F1 = zeros(6,1);

for i=1:10
    %divide into train, validation, test
    VI = find(Indices == i); %validation indices
    
    j = i + 1; 
    if j==11
        j = 1; 
    end
    
    TestI = find(Indices == j); %test indices
    
    TI = find(Indices ~= i & Indices ~= j); %train indices
    
    %define network and parameters
    net = feedforwardnet(topology); 
    net = configure(net,data,labels);
    net.numLayers = 3; 
    net.trainFcn = 'trainrp';
    net.trainParam.epochs = 3000;
    net.trainParam.lr = lr;

    %trainrp parameters
    net.trainParam.delt_inc = inc; 
    net.trainparam.delt_dec = dec; 

    net.performParam.regularization = 0.1;
    net.divideFcn = 'divideind'; % Divide data by indices (i.e. not randomly)
    net.divideParam.trainInd = TI;
    net.divideParam.valInd = VI;
    net.divideParam.testInd = TestI;
    net.outputs{2}.processFcns= {}; 
    
    net = train(net,data,labels); %train network

    % Test the Network
    test_labels = transpose(y(TestI,:));
    outputs = net(data(:,TestI));
    [M,classes] = max(outputs,[],1);
    err = size(find((test_labels-classes)~=0),2) / size(classes,2); 
    performance = perform(net,labels(:,TestI),outputs);
    
    if (err < BestErr)
        BestErr = err;
        ErrNet = net;
        ErrInd = i;
    end
    if (performance < BestPerf)
        BestPerf = performance;
        PerfNet = net;
        PerfInd = i;
    end
    
    predictions = testANN(net,data(:,TestI)');
    [undefined, predictions] = ANNdata(0, predictions); 
    confSum =confSum + confusionMatrixGeneration(predictions', y(TestI,:)'); %confusion matrix for each fold
end
    confusionMatrix = confSum./10   %average confusion matrix

    a = 1; %equal weight for recall and precision in F1

    %calculate recall/precision/F1 for each emotion using confusion matrix
    for i=1:6
        recall(i) = 100*confusionMatrix(i,i)/sum(confusionMatrix(i, :));
        precision(i) = 100*confusionMatrix(i,i)/sum(confusionMatrix(:,i));
        F1(i) = (1+a)*(recall(i)*precision(i))/(a*precision(i)+recall(i)); 
    end

end