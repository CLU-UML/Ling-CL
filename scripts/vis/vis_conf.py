import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])
df = df[df.experiment != '--']

sns.lineplot(data = df, x = 'step', y = 'value', color = 'hparams.curr', style = '
