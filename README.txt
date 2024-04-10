File execution instructions for logistic regression.

We need to build our vocabulary with the most common words. To do this we open the data.py file to edit some data.
    i) In the __init__ function give mostCommonWordsToKeep a number for the most common words we want to keep. (There is a relevant comment next to the command).
    ii) In the __init__ function give mostCommonWordsToDiscard a number that will indicate how many of the first most common words we have already selected in (i) we will discard according to the pronunciation (There is a relevant comment next to the command).
    Sub-queries (i) and (ii) have some default values which you can not change if you don't want to.
    iii) In the link function, give the url the path where the folder with the data from Imdb is located exactly as it is in the predefined value of the url. Pay attention that for it to work you must put a double slash (\\) after each folder. This step is mandatory to run the file on your computer.

Note1: Each py file takes some minutes to run (Especially if we use a large number of words that we want to keep).
Note2: If an error occurs when executing logistic.py from cmd, run it from idle.
