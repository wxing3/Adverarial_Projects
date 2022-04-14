function precision = computePrecision(trueTags, predictedAction)
    numOfPredictedBits = sum(predictedAction);
    if(numOfPredictedBits == 0)
        precision = 0;
    else
        precision = (trueTags' * predictedAction) / numOfPredictedBits;
    end
end
