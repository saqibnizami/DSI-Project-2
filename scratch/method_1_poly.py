# Get data
t = pd.read_csv("train.csv")
tt = pd.read_csv("test.csv")
t.columns,tt.columns
len(yt)

t.drop(['Id',], axis=1, inplace=True)
tt.drop('Id', axis=1, inplace=True)
t.shape,tt.shape

train = pd.concat((t,tt)).reset_index(drop=True) 

train['SalePrice'].isnull().sum()
y = pd.DataFrame(train['SalePrice'])
y.shape

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
y= imp.fit_transform(y)

y.shape



obj_ranker(X)

X[rankcol]



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


# X[over50p].isnull().sum()
# X[over50p] = X[over50p].fillna(method='median')
imp = Imputer(strategy='median', axis=0)
X[over50p] = imp.fit_transform(X[over50p])


forpoly = X[over50p]
rest_of_X = X.drop(columns=over50p)



poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
polyx = pd.DataFrame(poly.fit_transform(forpoly), columns=poly.get_feature_names(forpoly.columns))



X = polyx.join(rest_of_X, how='left')


catfeats = X.select_dtypes(include = ["object"]).columns
numfeats = X.select_dtypes(exclude = ["object"]).columns
numfeats = numfeats.drop("SalePrice")
print("Numerical features : " + str(len(numfeats)))
print("Categorical features : " + str(len(catfeats)))
Xnum = X[numfeats]
Xcat = X[catfeats]


print("NAs for numerical features in X : " + str(Xnum.isnull().values.sum()))
Xnum = Xnum.fillna(Xnum.median())
print("Remaining NAs for numerical features in X : " + str(Xnum.isnull().values.sum()))

skewness = Xnum.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed = skewness.index
Xnum[skewed] = np.log1p(Xnum[skewed])

print("NAs for categorical features in X : " + str(Xcat.isnull().values.sum()))
Xcat = pd.get_dummies(Xcat)
print("Remaining NAs for categorical features in X : " + str(Xcat.isnull().values.sum()))

X = pd.concat([Xnum, Xcat], axis = 1)
print("New number of features : " + str(X.shape[1]))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y,random_state = 19)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

ss = StandardScaler()
X_train.loc[:, numfeats] = ss.fit_transform(X_train.loc[:, numfeats])
X_test.loc[:, numfeats] = ss.transform(X_test.loc[:, numfeats])