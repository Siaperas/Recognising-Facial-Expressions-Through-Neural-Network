function [confMatrix] = confusionMatrixGeneration(predictions, targets)

confMatrix = zeros(6,6); 

for i=1:size(predictions,1)
    ind = find(predictions(i,:)); 
    if (ind == targets(i))
        confMatrix(ind,ind) = confMatrix(ind,ind) + 1; 
    else
        confMatrix(targets(i), ind) = confMatrix(targets(i), ind) + 1; 
    end
end

    