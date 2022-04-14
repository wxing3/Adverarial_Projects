function f1Score = computeF1(action1, action2)
    numOfTotalOneBits = sum(action1) + sum(action2);
    if(numOfTotalOneBits == 0)
        f1Score = 1;
    else
        f1Score = 2 * (action1' * action2) / numOfTotalOneBits;
    end
end
