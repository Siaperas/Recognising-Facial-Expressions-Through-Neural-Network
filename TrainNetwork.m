function [BestPerf,BestErr,ErrNet,PerfNet,ErrInd,PerfInd] = TrainNetwork(data,y,  topology,lr,lrinc,lrdec,mc,inc,dec)

data = transpose(data);
[labels] = mapToTargets(y);
labels = transpose(labels);
Indices = importdata('clean_indices.mat');
BestPerf = 1e10;
BestErr = size(data,1);

for i=1:10
    VI = find(Indices == i);
    TI = find(Indices ~= i);
    net = feedforwardnet(topology); 
    net = configure(net,data,labels);
    net.numLayers = 3; 
    net.trainFcn = 'trainrp';
    net.trainParam.epochs = 3000;
    net.trainParam.lr = lr;
      
     %for traingda
     % net.trainParam.lr_inc = lrinc;
     % net.trainParam.lr_dec = lrdec;

     %for traingdm
     % net.trainParam.mc = mc; 

    %for trainrp
    net.trainParam.delt_inc = inc; 
    net.trainparam.delt_dec = dec; 

    net.performParam.regularization = 0.1;
    net.divideFcn = 'divideind'; % Divide data by indices (i.e. not randomly)
    net.divideParam.trainInd = TI;
    net.divideParam.valInd = VI;
    net.divideParam.testInd = [];
    net.outputs{2}.processFcns= {}; 

    net = train(net,data,labels);

    % Test the Network
    test_labels = transpose(y(VI,:));
    outputs = net(data(:,VI));
    [M,classes] = max(outputs,[],1);
    err = size(find((test_labels-classes)~=0),2) / size(classes,2); 
    performance = perform(net,labels(:,VI),outputs);
    
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

end

end