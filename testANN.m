function [predictions] = testANN(net,data)

data = transpose(data);

outputs = net(data);

predictions = NNout2labels(outputs);

end