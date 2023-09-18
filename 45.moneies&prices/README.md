# 2. Tehran House Price
* 5 most expensive houses in Tehran

|               |       Area     |       Room     |       Parking     |       Warehouse     |      Elevator     |      Address     |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1      | 420 | 4 | have | have | have | Zaferanieh || ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 2      | 705 | 5 | have | have | don`t have | Abazar || ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 3      | 400 | 5 | have | have | don`t have | Lavasan || ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 4      | 680 | 5 | have | have | don`t have | Ekhtiarieh || ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 5      | 350 | 4 | have | have | have | Niavaran |

* Compare my model with Scikit-Learn

|        Model       |       MAE     |       MSE     |      RMSE     |
| ------------- | ------------- | ------------- | ------------- |
| My LLS Model  | 29177674588.87284 | 4.6788358666096837e+20 | 21630616881.193386 || ------------- | ------------- | ------------- | ------------- |
| Sklearn LinearRegression    | 3006285642.9841785 | 3.519317809904408e+19 | 5932383846.232818 || ------------- | ------------- | ------------- | ------------- |
| Sklearn RidgeCV     | 3001949151.169838 | 3.520814321517738e+19 | 5933645019.30958 |
___
### Why the MSE metric is a very large number?
* Because the numbers of the differences have been raised to the power of 2 to get out of the negative state, but at the end, no root has been taken from them.

# $ vs. ريال

* Show the highest dollar and the lowest price in Ahmadinejad, Rouhani and Raisi's presidency respectively

| President | Highest Price | Lowest Price|
| ------------- | ------------- | ------------- |
| Ahmadinejad | 39,700 | 13,350 || ------------- | ------------- | ------------- | ------------- |
| Rouhani | 320,060 | 12,850 || ------------- | ------------- | ------------- | ------------- |
| Raiesi | 555,600 | 251,250 |

### Evaluate each model on test dataset using MAE loss function in Scikit-Learn

* Ahmadinejad data : 10109.422194809302
* Rouhani data : 160980.70424317414
* Raiesi data : 193136.85650221317