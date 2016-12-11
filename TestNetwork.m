function [Err,Perf] = TestNetwork(net,data,y)

data = transpose(data);
[labels] = mapToTargets(y);
labels = transpose(labels);

outputs = net(data);
[M,classes] = max(outputs,[],1);
Err = size(find((transpose(y)-classes)~=0),2) / size(classes,2); 
Perf = perform(net,labels,outputs);

end