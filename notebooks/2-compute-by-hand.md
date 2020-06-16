---
title:  'Computing regression coefficients'
author: 'Fraida Fund'

---

::: {.cell .markdown}

## Computing regression coefficients

This notebook walks through the steps involved in manual (pen and paper) computation of the optimal least squares regression coefficients.

We'll also show how to do each of these steps in Python, so that you can try other values for $X$ and $y$ and then check your work.

:::

::: {.cell .code}
```python
import pandas as pd
import seaborn as sns
import numpy as np

%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```
:::



::: {.cell .markdown}

We will start with the labeled data. Our training data includes four samples, with two features:

$$ X = 
\begin{bmatrix}
1 & 2 \\
2 & 3 \\
4 & 1 \\
5 & 5 
\end{bmatrix},
y = 
\begin{bmatrix}
3  \\
2 \\
7 \\
1 
\end{bmatrix}
$$

:::


::: {.cell .code}
```python
X = [[1,2], [2, 3], [4, 1], [5, 5]]
y = [3, 2, 7, 1]
```
:::



::: {.cell .markdown}

We can use `seaborn` to plot each column of $X$ versus $y$:

:::


::: {.cell .code}
```python
df = pd.DataFrame(X, columns = [ 'x1', 'x2'])
df = df.assign(y=y)

melted = df.melt(id_vars=['y'], value_vars=[ 'x1','x2'])
g = sns.FacetGrid(melted, col='variable', col_wrap=3);
g.map(sns.scatterplot, 'value', 'y');
```
:::

::: {.cell .markdown}

Next, we will create a matrix $A$ by horizontally “stacking” a vector
of 1s and the matrix $X$:

$$ A = 
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 3 \\
1 & 4 & 1 \\
1 & 5 & 5 
\end{bmatrix},
$$


This will enable us to also compute the intercept, $\beta_0$, using the same matrix calculation as we use to compute $\beta_1$ and $\beta_2$.

:::


::: {.cell .code}
```python
A = np.hstack((np.ones(4)[:,None], X))
A
```
:::

::: {.cell .markdown}

The optimal least squares values for the vector $\beta$ are

$$\beta^* = (A^T A)^{-1} A^{T} y$$


Note that the least-squares solutions are the solutions of the matrix equation

$$ A^T A \beta = A^T y$$

A matrix equation is in the form $Ax=b$, where $A$ is an $m \times n$ matrix and $b$ is a column vector with $m$ entries. It can be solved for $x$ by forming the augmented matrix $(A | b)$ and then using elementary row operations to get it in row reduced form.


Thus, to get $\beta$ we will:

* Compute the matrix $A^T A$ and the vector $A^T y$.
* Form the augmented matrix for the matrix equation $A^T A \beta = A^T y$: $(A^T A | A^T y)$
* Row reduce to find the estimate for $\beta$.

:::


::: {.cell .markdown}

The transpose of $A$, $A^T$, is the matrix whose rows are the columns of $A$:



$$ A^T = 
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 2 & 4 & 5 \\
2 & 3 & 1 & 5 
\end{bmatrix}
$$

:::


::: {.cell .code}
```python
A.T
```
:::


::: {.cell .markdown}

To solve

$$ A^T A \beta = A^T y$$

we'll need $A^T A$:

$$ A^T A = 
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 2 & 4 & 5 \\
2 & 3 & 1 & 5 
\end{bmatrix}
\begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 3 \\
1 & 4 & 1 \\
1 & 5 & 5 
\end{bmatrix} = 
\begin{bmatrix}
4 & 12 & 11 \\
12 & 46 & 37 \\
11 & 37 & 39 
\end{bmatrix}
$$

:::


::: {.cell .code}
```python
A.T.dot(A)
```
:::


::: {.cell .markdown}

and $A^T y$:


$$ A^T y = 
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 2 & 4 & 5 \\
2 & 3 & 1 & 5 
\end{bmatrix}
\begin{bmatrix}
3  \\
2 \\
7 \\
1 
\end{bmatrix} =
\begin{bmatrix}
13 \\
40 \\
24 
\end{bmatrix}
$$

:::


::: {.cell .code}
```python
A.T.dot(y)
```
:::



::: {.cell .markdown}

Next, create the augmented matrix $(A^T A | A^T y)$:

$$ (A^T A | A^T y)=
\begin{bmatrix}
4  & 12 & 11 & | & 13 \\
12 & 46 & 37 & | & 40 \\
11 & 37 & 39 & | & 24
\end{bmatrix}
$$

:::


::: {.cell .code}
```python
M1 = np.hstack([A.T.dot(A), A.T.dot(y)[:,None]])
M1
```
:::

::: {.cell .markdown}

We will perform some elementary row operations on the augmented matrix to get $A^T A$ in row reduced form:

:::


::: {.cell .code}
```python
M2 = np.vstack([M1[0], M1[1]-3*M1[0], 4*M1[2]-11*M1[0]])
M2
```
:::


::: {.cell .code}
```python
M3 = np.vstack([M2[0], M2[1], 5*M2[2]-8*M2[1]])
M3
```
:::


::: {.cell .code}
```python
M4 = np.vstack([M3[0], 143*M3[1]-4*M3[2], M3[2]])
M4
```
:::


::: {.cell .code}
```python
M5 = np.vstack([143/2*(M4[0]-12/1430*M4[1] - 11/143*M4[2]), M4[1], M4[2]])
M5
```
:::


::: {.cell .code}
```python
M6 = np.vstack([M5[0]/286, M5[1]/1430, M5[2]/143])
M6
```
:::


::: {.cell .markdown}

When we're finished, our augmented matrix looks like this:

$$
\begin{bmatrix}
1 & 0 & 0 & | & \frac{1597}{286} \\
0 & 1 & 0 & | & \frac{1115}{1430} \\
0 & 0 & 1 & | & \frac{-243}{143}
\end{bmatrix}
$$

and so

$$ \beta^* = \left[ \frac{1597}{286}, \frac{1115}{1430}, \frac{-243}{143} \right] $$

:::


::: {.cell .code}
```python
M6[:,3]
```
:::

::: {.cell .markdown}

Compare this to the known solution:

:::


::: {.cell .code}
```python
np.linalg.inv( A.T.dot(A) ).dot(A.T.dot(y))
```
:::

