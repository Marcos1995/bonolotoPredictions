import sqliteClass
import pandas as pd

df = pd.read_csv("Lotoideas.com - Histórico de Resultados - Bonoloto - 2013 a 2022.csv")

print(df)

a = sqliteClass.db()