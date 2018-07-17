

# Load in 'train' and 'test' data.
train = pd.read_csv("train.csv")
print("Train set: \n", train.shape, "\n Columns: \n", train.columns)
test = pd.read_csv("test.csv")
print("test set: \n", test.shape, "\n Columns: \n", test.columns)

# SalePrice is our target
price = train['SalePrice']
# Natural log transformation of target, ascribe equal weight to errors 
# Ref: https://people.duke.edu/~rnau/411log.htm
y = np.log1p(price)

print("Differing columns: \n", [x for x in train.columns if x not in test.columns])
print("Duplicate entries: ", train.shape[0] - len(set(train['Id'])))    
# print("Duplicate IDs: \n", [x for x in train.Id if x in test.Id])

# Id column isn't needed
X = train.drop(columns='Id')

y.shape, X.shape, test.shape

# Data Clean-up and Feature Engineering Run

# The Ames, Iowa data dataset documentation states:
# > Potential Pitfalls (Outliers): Although all known errors were corrected in the data, no
# >observations have been removed due to unusual values and all final residential sales
# >from the initial data set are included in the data presented with this article. There are
# >five observations that an instructor may wish to remove from the data set before giving
# >it to students (a plot of SALE PRICE versus GR LIV AREA will quickly indicate these
# >points). Three of them are true outliers (Partial Sales that likely don’t represent actual
# >market values) and two of them are simply unusual sales (very large houses priced
# >relatively appropriately). I would recommend removing any houses with more than
# >4000 square feet from the data set (which eliminates these five unusual observations)
# >before assigning it to students. 
# >>Reference : https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
# >>Cock, D. De. (2011). Ames , Iowa : Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics >>Education, 19(3), 1–15. Retrieved from www.amstat.org/publications/jse/v19n3/decock.pdf

sns.regplot(X['Gr Liv Area'], X['SalePrice'])

# This dataset has the two outliers mentioned in the paper.

X = X[X['Gr Liv Area']<4000]

sns.regplot(X['Gr Liv Area'], X['SalePrice'])

# Take log of the target. Taking the log will mean that the errors in predicting
# expensive houses will affect the result equally as the errors in predicting cheap 
# houses. Ref: https://people.duke.edu/~rnau/411log.htm

# X['SalePrice'] = np.log1p(X['SalePrice'])
y = np.log1p(X['SalePrice'])

# DATADICT, lets see what objects we can clean up
datadict = {}
rankables = []
rankcol = []
for x in columntypes(X)['objects']:
        
    datadict.update({x : np.ndarray.tolist(X[x].unique())})
for key,value in sorted(datadict.items()):
    if key != 'PID':
        print(key, "\n", value, "\n")
        rankables.append([x for x in value])
        rankcol.append(key)

'''
Alley - NAN = NOT THERE, RANK
 [nan, 'Pave', 'Grvl'] 

Bldg Type 
 ['1Fam', 'TwnhsE', 'Twnhs', '2fmCon', 'Duplex'] 

Bsmt Cond - NAN = 0, RANK
 ['TA', 'Gd', nan, 'Fa', 'Po', 'Ex'] 

Bsmt Exposure - NAN = 0, RANK
 ['No', 'Gd', 'Av', nan, 'Mn'] 

Bsmt Qual - NAN = 0, RANK
 ['TA', 'Gd', 'Fa', nan, 'Ex', 'Po'] 

BsmtFin Type 1 - NAN = 0, COMBINE WITH TYPE 2, RANK
 ['GLQ', 'Unf', 'ALQ', 'Rec', nan, 'BLQ', 'LwQ'] 

BsmtFin Type 2 
 ['Unf', 'Rec', nan, 'BLQ', 'GLQ', 'LwQ', 'ALQ'] 

Central Air - BINARY!
 ['Y', 'N'] 

Condition 1 - COMBINE
 ['RRAe', 'Norm', 'PosA', 'Artery', 'Feedr', 'PosN', 'RRAn', 'RRNe', 'RRNn'] 

Condition 2 
 ['Norm', 'RRNn', 'Feedr', 'Artery', 'PosA', 'PosN', 'RRAe', 'RRAn'] 

Electrical 
 ['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'] 

Exter Cond - RANK 0 - 5
 ['TA', 'Gd', 'Fa', 'Ex', 'Po'] 

Exter Qual - RANK 0 - 5
 ['Gd', 'TA', 'Ex', 'Fa'] 

Exterior 1st - COMBINE
 ['HdBoard', 'VinylSd', 'Wd Sdng', 'BrkFace', 'Plywood', 'MetalSd', 'AsbShng', 'CemntBd', 'WdShing', 'Stucco', 'BrkComm', 'Stone', 'CBlock', 'ImStucc', 'AsphShn'] 

Exterior 2nd - COMBINE
 ['Plywood', 'VinylSd', 'Wd Sdng', 'HdBoard', 'MetalSd', 'AsbShng', 'CmentBd', 'Wd Shng', 'BrkFace', 'Stucco', 'Brk Cmn', 'ImStucc', 'Stone', 'CBlock', 'AsphShn'] 

Fence - NAN = NOTTHERE
 [nan, 'MnPrv', 'GdPrv', 'GdWo', 'MnWw'] 

Fireplace Qu - NAN = 0, RANK
 [nan, 'TA', 'Gd', 'Po', 'Ex', 'Fa'] 

Foundation 
 ['CBlock', 'PConc', 'BrkTil', 'Slab', 'Stone', 'Wood'] 

Functional RANK??
 ['Typ', 'Mod', 'Min2', 'Maj1', 'Min1', 'Sev', 'Sal', 'Maj2'] 

Garage Cond - NAN = 0, RANK
 ['TA', 'Fa', nan, 'Po', 'Gd', 'Ex'] 

Garage Finish - NAN = NOTTHERE
 ['RFn', 'Unf', 'Fin', nan] 

Garage Qual - NAN = 0, RANK
 ['TA', 'Fa', nan, 'Gd', 'Ex', 'Po'] 

Garage Type - - NAN = NOTTHERE
 ['Attchd', 'Detchd', 'BuiltIn', 'Basment', nan, '2Types', 'CarPort'] 

Heating 
 ['GasA', 'GasW', 'Grav', 'Wall', 'OthW'] 

Heating QC - RANK
 ['Ex', 'TA', 'Gd', 'Fa', 'Po'] 

House Style 
 ['2Story', '1Story', '1.5Fin', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin', '1.5Unf'] 

Kitchen Qual - RANK
 ['Gd', 'TA', 'Fa', 'Ex'] 

Land Contour 
 ['Lvl', 'HLS', 'Bnk', 'Low'] 

Land Slope RANK
 ['Gtl', 'Sev', 'Mod'] 

Lot Config 
 ['CulDSac', 'Inside', 'Corner', 'FR2', 'FR3'] 

Lot Shape RANK
 ['IR1', 'Reg', 'IR2', 'IR3'] 

MS Zoning 
 ['RL', 'RM', 'FV', 'C (all)', 'A (agr)', 'RH', 'I (all)'] 

Mas Vnr Type - NAN = NONE
 ['BrkFace', 'None', nan, 'Stone', 'BrkCmn'] 

Misc Feature - - NAN = NOTTHERE
 [nan, 'Shed', 'TenC', 'Gar2', 'Othr', 'Elev'] 

Neighborhood 
 ['Sawyer', 'SawyerW', 'NAmes', 'Timber', 'Edwards', 'OldTown', 'BrDale', 'CollgCr', 'Somerst', 'Mitchel', 'StoneBr', 'NridgHt', 'Gilbert', 'Crawfor', 'IDOTRR', 'NWAmes', 'Veenker', 'MeadowV', 'SWISU', 'NoRidge', 'ClearCr', 'Blmngtn', 'BrkSide', 'NPkVill', 'Blueste', 'GrnHill', 'Greens', 'Landmrk'] 

Paved Drive - RANK, 0, 1, 2
 ['Y', 'N', 'P'] 

Pool QC - - NAN = 0, RANK
 [nan, 'Fa', 'Gd', 'Ex', 'TA'] 

Roof Matl 
 ['CompShg', 'WdShngl', 'Tar&Grv', 'WdShake', 'Membran', 'ClyTile'] 

Roof Style 
 ['Gable', 'Hip', 'Flat', 'Mansard', 'Shed', 'Gambrel'] 

Sale Type 
 ['WD ', 'New', 'COD', 'ConLD', 'Con', 'CWD', 'Oth', 'ConLI', 'ConLw'] 

Street - RANK
 ['Pave', 'Grvl'] 

Utilities RANK
 ['AllPub', 'NoSeWa', 'NoSewr'] 
'''

def obj_ranker(df):

    objects = columntypes(df)['objects']
    df[objects] = df[objects].replace(["No", "N"], 0)

    df[objects] = df[objects].replace(["P", "Grvl", "Po", "Mn", "Unf",
                                       "Sal", "Sev", "IR3"], 1)
    df[objects] = df[objects].replace(["Y", "Fa", "Pave", "LwQ", "IR2",
                                       "NoSeWa"], 2)

    df[objects] = df[objects].replace(["Av", "TA", "Rec", "Maj2", "Gtl",
                                       "IR1", "NoSewr"], 3)

    df[objects] = df[objects].replace(["Gd", "AllPub", "Reg", "Maj1", "BLQ"], 4)

    df[objects] = df[objects].replace(["ALQ", "Mod", "Ex"], 5)

    df[objects] = df[objects].replace(["Min2","GLQ"], 6)
    df[objects] = df[objects].replace("Min1", 7)
    df[objects] = df[objects].replace("Typ", 8)
    
    for x in rankcol:
        df[x] = df[x].fillna(0)
    

obj_ranker(X)

X[rankcol]

ranks = ["P", "Grvl", "Po", "Mn", "Unf", "Sal", "Sev", "IR3","Y", "Fa", "Pave", "LwQ", "IR2", "NoSeWa","Av", "TA", "Rec", "Maj2", "Gtl", "IR1", "NoSewr","Gd", "AllPub", "Reg", "Maj1", "BLQ","ALQ", "Mod", "Ex"]


# Make MetaScores by combining ranked features


X["Overall"] = X["Overall Qual"] * X["Overall Cond"]
X["External"] = X["Exter Qual"] * X["Exter Cond"]
X["Garage"] = X["Garage Qual"] * X["Garage Cond"] 
X["Garage Area Qual"] = 2*(X["Garage Area"] + X["Garage Qual"])
X["Kitchen"] = X["Kitchen AbvGr"] * X["Kitchen Qual"]
X["Fireplace"] = X["Fireplaces"] * X["Fireplace Qu"]
X["Pool"] = X["Pool Area"] * X["Pool QC"]

X["Total SqFt"] = X["Gr Liv Area"] + X["Total Bsmt SF"]
X["Abv Grade SqFt"] = X["1st Flr SF"] + X["2nd Flr SF"]
X["Porch SqFt"] = (X["Open Porch SF"] + X["Enclosed Porch"] + 
                   X["3Ssn Porch"] + X["Screen Porch"])

X["Total Bath Num"] = (X["Bsmt Full Bath"] + (0.5 * X["Bsmt Half Bath"]) +
                       X["Full Bath"] + (0.5 * X["Half Bath"]))




corr = X.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corrs = pd.DataFrame(corr.SalePrice)
over50 = corrs.iloc[1:11,:].index
over50p = [x for x in over50]
print(over50p)

# X[over50p].isnull().sum()
# X[over50p] = X[over50p].fillna(method='median')
imp = Imputer(strategy='median', axis=0)
X[over50p] = imp.fit_transform(X[over50p])

X[over50p].isnull().sum()

forpoly = X[over50p]
rest_of_X = X.drop(columns=over50p)

rest_of_X.shape, forpoly.shape

forpoly.isnull().sum()

poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
polyx = pd.DataFrame(poly.fit_transform(forpoly), columns=poly.get_feature_names(forpoly.columns))


poly.get_feature_names(forpoly.columns)

polyx.shape

X = polyx.join(rest_of_X, how='left')

X.shape

catfeats = X.select_dtypes(include = ["object"]).columns
numfeats = X.select_dtypes(exclude = ["object"]).columns
numfeats = numfeats.drop("SalePrice")
print("Numerical features : " + str(len(numfeats)))
print("Categorical features : " + str(len(catfeats)))
Xnum = X[numfeats]
Xcat = X[catfeats]

Xnum.shape, Xcat.shape

print("NAs for numerical features in X : " + str(Xnum.isnull().values.sum()))
Xnum = Xnum.fillna(Xnum.median())
print("Remaining NAs for numerical features in X : " + str(Xnum.isnull().values.sum()))

# Log transform of the skewed numerical features to lessen impact of outliers
# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
skewness = Xnum.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed = skewness.index
Xnum[skewed] = np.log1p(Xnum[skewed])

print("NAs for categorical features in X : " + str(Xcat.isnull().values.sum()))
Xcat = pd.get_dummies(Xcat)
print("Remaining NAs for categorical features in X : " + str(Xcat.isnull().values.sum()))

# Join categorical and numerical features
X = pd.concat([Xnum, Xcat], axis = 1)
print("New number of features : " + str(X.shape[1]))

y.shape

# Partition the dataset in train + validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 19)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

ss = StandardScaler()
X_train.loc[:, numfeats] = ss.fit_transform(X_train.loc[:, numfeats])
X_test.loc[:, numfeats] = ss.transform(X_test.loc[:, numfeats])

scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)

**2* Linear Regression with Ridge regularization (L2 penalty)**

From the *Python Machine Learning* book by Sebastian Raschka :  Regularization is a very useful method to handle collinearity, filter out noise from data, and eventually prevent overfitting. The concept behind regularization is to introduce additional information (bias) to penalize extreme parameter weights.  

Ridge regression is an L2 penalized model where we simply add the squared sum of the weights to our cost function.

def plotter(model, y_train_m, y_test_m, model_coef_ ):
    '''
    model is a string, name of the regression
    y_train_m is the model train prediction on X_train
    y_test_m is the model test prediction on X_test
    model_coef_ is the call for model coefs
    '''
    
    # Plot residuals
    plt.scatter(y_train_m, y_train_m - y_train, c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_test_m, y_test_m - y_test, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title(f"Linear regression with {model} regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc = "upper left")
    plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
    plt.show()
    
    # Plot predictions
    plt.scatter(y_train_m, y_train, c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_test_m, y_test, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title(f"Linear regression with {model} regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
    plt.show()
    
    # Plot important coefficients
    coefs = pd.Series(model_coef_, index = X_train.columns)
    print(f"{model} picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                         coefs.sort_values().tail(10)])
    imp_coefs.plot(kind = "barh")
    plt.title(f"Coefficients in the {model} Model")
    plt.show()

# 2* Ridge
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)

plotter("Ridge", y_train_rdg, y_test_rdg, ridge.coef_)

We're getting a much better RMSE result now that we've added regularization. The very small difference between training and test results indicate that we eliminated most of the overfitting. Visually, the graphs seem to confirm that idea.  

Ridge used almost all of the existing features.

**3* Linear Regression with Lasso regularization (L1 penalty)**

LASSO stands for *Least Absolute Shrinkage and Selection Operator*. It is an alternative regularization method, where we simply replace the square of the weights by the sum of the absolute value of the weights. In contrast to L2 regularization, L1 regularization yields sparse feature vectors : most feature weights will be zero. Sparsity can be useful in practice if we have a high dimensional dataset with many features that are irrelevant.  

We can suspect that it should be more efficient than Ridge here.

y_test.index

# 3* Lasso
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
print("Lasso RMSE on Test set :", rmse_cv_test(lasso))
y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)
predictions = np.array(y_test_las)

plotter("Lasso + CV", y_train_las, y_test_las, lasso.coef_)

# rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))
submitcsv = pd.DataFrame({'ID':y_test.index, 'SalePrice':y_test_las})
fileName = "submission.csv"
submitcsv.to_csv(fileName, index=False)





# 3* Lasso
lassolars = LassoLarsCV(fit_intercept=False, cv = 3, max_iter=lassolars.max_iter
, max_n_alphas=lassolars.max_n_alphas)
lassolars.fit(X_train, y_train)
alpha = lassolars.alpha_
print("Best alpha :", alpha)

# print("Try again for more precision with alphas centered around " + str(alpha))
# lasso = LassoLars(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
#                           alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
#                           alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
#                           alpha * 1.4], 
#                 max_iter = 50000, cv = 10)
# lassolars.fit(X_train, y_train)
# alpha = lasso.alpha_
# print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lassolars).mean())
print("Lasso RMSE on Test set :", rmse_cv_test(lassolars))
y_train_laslars = lassolars.predict(X_train)
y_test_laslars = lassolars.predict(X_test)
predictions = np.array(y_test_laslars)

plotter("LassoLarsCV", y_train_laslars, y_test_laslars, lassolars.coef_)
# rmse = math.sqrt(np.mean((np.array(y_test) - predictions)**2))
submitcsv = pd.DataFrame({'ID':y_test.index, 'SalePrice':y_test_laslars})
fileName = "submissionLARS.csv"
submitcsv.to_csv(fileName, index=False)




