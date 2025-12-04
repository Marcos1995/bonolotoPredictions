import mainClass

d = mainClass.predictData(
    dbFileName="predictions.sqlite",
    datasetTable="raffleDataset",
    predictionsTable="rafflePredictions",
    batch_size=1,
    epoch=10
)