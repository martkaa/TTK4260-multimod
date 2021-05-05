# Assignment 5

## Part 1: PCA plot with weight = 1 and random cross validation
This is the result of plotting data from windsolarpower data from 2016 with weight = 1 and random cross validation.

### Scores and loadings:

The scores and loadings are presented below. The loading plot describes how much a variable is weighted when it comes to describing as much of the variance as possible. From the plot one can see that most of the variance lays in the RadSurface and RadTop variables. The weight = 1 resulted in the variables with the largest values dominating the analysis. This is confirmed by looking at the columns for these variables in the data set.

![loadings.](Figures/Part_1/loadings.png)

The scores plot shows the weighting of the different samples. The correlation between the data points in the scores plot and loadings plot yields that data points in the upper left corner in scores wil have high values in RadSurface. Likewise, the data in the lower left hand corner will have high values in the RaTop variable. 

![scores.](Figures/Part_1/scores.png)

When adding grouping based on different months, a clear pattern is presented. The summer months with high radiation is represented with green on the left side, the large values in RadSurface and RadTop. Furthermore, the winter months are prepresented to the right.

![scores with grouping.](Figures/Part_1/scores_month.png)

### Explained variance:
The plot below shows the explained variance. One can see that PC1 explaines 96% of the variance alone, and by including PC2 one can describe 99% of the variance, only by looking at two dimentions of the data. 

![Explained variance.](Figures/Part_1/explained_variance.png)

### Correlation loadings
The correlation loadings plot shows the correlation between the score vectors and individual variables. It shows how much of the total variance of a variable is explained by the PCs. If a variable lies on the outer circle in the plot, 100% of its variance is explained by the PC. If it lies in the inner circle, 50% is explained. 

The plot shows that RadSurface, RadTop, IrrDiffuse and IrrDirect is nearly 100% explained by the PCs. The rest of the variables' variance is explained by under 50%.

![Correlation loadings.](Figures/Part_1/correlation_loadings.png)

## Part 2: PCA plot with weight = 1/std and random cross validation

![Loadings.](Figures/Part_2/Loadings.png)

![Scores.](Figures/Part_2/Scores.png)

![Scores grouped.](Figures/Part_2/scores_months.png)

### Explained variance:
![Explained variance.](Figures/Part_2/explained_variance.png)
### Correlation loadings:
> need correlation loadings with ellipse

### Optimal number of PCAs:


### Line Plot of Scores:

![Correlation loadings.](Figures/Part_2/daily_scores.png)


