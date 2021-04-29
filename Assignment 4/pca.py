import pandas as pd
import numpy as np 
import random as rd 
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Create sample data set

genes = ['gene' + str(i) for i in range(1, 101)]
wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]

data = pd.DataFrame(columns = [*wt, *ko], index = genes) # * pakker ut arrays

for gene in data.index:
    data.loc[gene, 'wt1': 'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    data.loc[gene, 'ko1': 'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
print(data.head())
print(data.shape)

# Må skalere før PCA. Transponerer data fordi scale() forventer rader istede for kolonner. 

scaled_data = preprocessing.scale(data.T) 
# StandardScaler().fir_transform(data.T)

pca = PCA(.95) # minimal number pcas such that 95% of variance is retained
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1) # calculate the percentage of variation that eacj principal comonent accounts for
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] # create labels for the scree plot. one label per principal comonent i.e PC1

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Comonent')
plt.title('Scree plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index = [*wt,*ko], columns = labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA graph')
plt.ylabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC1 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
plt.show()

loading_scores = pd.Series(pca.components_[0], index = genes) # series object with loading scores of PC1
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False) # sort loading scores based on their magnitude (abs)
top_10_genes = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_genes])