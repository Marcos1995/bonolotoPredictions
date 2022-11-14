import sqliteClass
import pandas as pd
import datetime as dt

df = pd.read_csv("Lotoideas.com - Histórico de Resultados - Bonoloto - 2013 a 2022.csv")

print(df.info)

df["FECHA"] = pd.to_datetime(df["FECHA"]).dt.date

print(df.head(10))

a = sqliteClass.db()

a.insertIntoFromPandasDf(sourceDf=df, targetTable="bonolotoResults")