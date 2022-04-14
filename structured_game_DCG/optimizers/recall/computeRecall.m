function recall = computeRecall(trueTags, predictedAction)
    numOfTrueTagBits = sum(trueTags);
    if(numOfTrueTagBits == 0)
        recall = 0;
    else
        recall = (trueTags' * predictedAction) / numOfTrueTagBits;
    end
end
