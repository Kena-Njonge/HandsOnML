# Paul Graham: Better Bayesian Filtering 2003
Naive Bayes performs well if the dataset is large enough, 99.5% accuracy with only .03% false positives.

Message headers should not be ignored.

Stemming of root words e.g. mailing to mail, may be premature optimizaiton. 

Limiting the length of the tokens that you use for the probability calculation can also be helpful e.g. only using the 15 most significant tokens, this also prevents spoofing. 

Biasing against false postives is a reasonable way to go, we would rather misclassify positive instances (classify ham as spam) instead of  misclassify negative instances (classigy spam as ham). This has implifications for the recall/accuracy tradeoff, we would rather have higher accuracy and lower recall.

Actually, even better this is a perfect use-case for the ROC curve as that plots the recall agaisnt the false positive rate and we would like to minimize the false positive rate. 

We should use any inherent metadata that we have for the classification and not treat spam classification as a pure text-classification problem.

Paul Graham uses the following more complicated definition of tokens for his model.

    Now I have a more complicated definition of a token:
    Case is preserved.

    Exclamation points are constituent characters.

    Periods and commas are constituents if they occur between two digits. This lets me get ip addresses and prices intact.

    A price range like $20-25 yields two tokens, $20 and $25.

    Tokens that occur within the To, From, Subject, and Return-Path lines, or within urls, get marked accordingly. E.g. ``foo'' in the Subject line becomes ``Subject*foo''. (The asterisk could be any character you don't allow as a constituent.)


The downside of the above characterisation is that you increase the filter's vocabulary which can mean that you have higher miss rate for unseen data e.g. 'free!!' may be classified as havign a 99% spam probability, while 'free!!!' as it is useen wouldn't have any associated probability.

Look at links and images, possibly ignore all other html.

The spam of the future (today) has correctly evolved like Paul Graham said it would into mostly neutral text followed by a link e.g. 'Your invoice from ...'

