models={
    "linear regression": LinearRegression(),
    "lasso": Lasso(),
    "ridge": Ridge(),
    "knn": KNeighborsRegressor(),
    "dt": DecisionTreeRegressor(),
    "rfr": RandomForestRegressor(),
    "xg":XGBRegressor(),
    "cat": CatBoostRegressor(verbose=False),
    "AdaBoost":AdaBoostRegressor()
}
