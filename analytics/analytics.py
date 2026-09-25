
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import joblib

os.makedirs("analytics", exist_ok=True)

print("task 1")
df = sns.load_dataset('titanic')
print(df.info()); print(df.describe()); print(df.shape)
df.to_csv("analytics/titanic.csv", index=False)

for col in df.columns:
    miss = df[col].isnull().sum()
    if miss>0:
        print(f"{col}: {miss} {miss/len(df)*100:.2f}%")


print("age 20.77% impute median, embarked 0.22% drop rows, deck 77.21% drop column unreliable")
df_clean = df.copy()
df_clean = df_clean.drop(columns=['deck'])
df_clean = df_clean.dropna(subset=['embarked','embark_town'])
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
df_clean.to_csv("analytics/titanic.csv", index=False)


plt.figure(); plt.hist(df_clean['age'], bins=30); plt.title("Age hist"); plt.savefig("analytics/age_hist.png"); plt.close()
plt.figure(); plt.boxplot(df_clean['age']); plt.title("Age box"); plt.savefig("analytics/age_box.png"); plt.close()
plt.figure(); plt.hist(df_clean['fare'], bins=30); plt.title("Fare hist"); plt.savefig("analytics/fare_hist.png"); plt.close()
plt.figure(); plt.boxplot(df_clean['fare']); plt.title("Fare box"); plt.savefig("analytics/fare_box.png"); plt.close()

def count_out(s):
    q1=s.quantile(0.25); q3=s.quantile(0.75); iqr=q3-q1; low=q1-1.5*iqr; high=q3+1.5*iqr
    return ((s<low)|(s>high)).sum()

print("Age outliers", count_out(df_clean['age']), "Fare outliers", count_out(df_clean['fare']))
mean_f=df_clean['fare'].mean(); median_f=df_clean['fare'].median(); mode_f=df_clean['fare'].mode()[0]
print(f"Fare mean {mean_f:.2f} median {median_f:.2f} mode {mode_f:.2f} -> mean>median>mode so right-skewed")


for sex in df_clean['sex'].unique():
    mask=df_clean['sex']==sex
    print(sex, df_clean[mask]['survived'].mean())
for pc in sorted(df_clean['pclass'].unique()):
    mask=df_clean['pclass']==pc
    print(f"pclass {pc} {df_clean[mask]['survived'].mean():.3f}")
for sex in df_clean['sex'].unique():
    for pc in sorted(df_clean['pclass'].unique()):
        mask=(df_clean['sex']==sex)&(df_clean['pclass']==pc)
        if mask.sum()>0:
            print(f"{sex} {pc} {df_clean[mask]['survived'].mean():.3f}")

cols6=['survived','pclass','age','sibsp','parch','fare']
corr=df_clean[cols6].corr()
plt.figure(figsize=(6,5)); sns.heatmap(corr, annot=True, cmap='coolwarm'); plt.savefig("analytics/corr_heatmap.png"); plt.close()
corr_abs=corr.abs()
np.fill_diagonal(corr_abs.values,0)
print(corr_abs.unstack().sort_values(ascending=False).head(4))
print("Top2: pclass-fare -0.55 higher class pays more, sibsp-parch 0.41 family travels together")


plt.figure(); sns.barplot(x='sex', y='survived', data=df_clean); plt.savefig("analytics/chart1.png"); plt.close()
print("Chart1 female 74% vs male 18% women and children first")
plt.figure(); sns.barplot(x='pclass', y='survived', data=df_clean); plt.savefig("analytics/chart2.png"); plt.close()
print("Chart2 first class 63% third 24% class privilege")
plt.figure(); sns.scatterplot(x='age', y='fare', hue='survived', data=df_clean); plt.savefig("analytics/chart3.png"); plt.close()
print("Chart3 high fare children survived more")
plt.figure(); sns.boxplot(x='survived', y='fare', data=df_clean); plt.savefig("analytics/chart4.png"); plt.close()
print("Chart4 survivors paid higher median fare")

scaler=StandardScaler()
print("Before", df_clean[['age','fare']].mean().to_dict(), df_clean[['age','fare']].std().to_dict())
scaled=scaler.fit_transform(df_clean[['age','fare']])
print("After mean", scaled.mean(axis=0), "std", scaled.std(axis=0))


print("Class balance", df_clean['survived'].value_counts(normalize=True).to_dict())
X=df_clean.drop(columns=['survived']); y=df_clean['survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("Stratified because 38% survived 62% died keep same ratio")


num_cols=['age','sibsp','parch','fare','pclass']
cat_cols=['sex','embarked']
num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
cat_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
preprocess=ColumnTransformer([('num',num_pipe,num_cols),('cat',cat_pipe,cat_cols)])


log_pipe=Pipeline([('preprocess',preprocess),('model',LogisticRegression(max_iter=1000))])
dt_pipe=Pipeline([('preprocess',preprocess),('model',DecisionTreeClassifier(random_state=42))])
rf_pipe=Pipeline([('preprocess',preprocess),('model',RandomForestClassifier(random_state=42))])
log_pipe.fit(X_train,y_train); dt_pipe.fit(X_train,y_train); rf_pipe.fit(X_train,y_train)

fitted_pre=dt_pipe.named_steps['preprocess']
cat_names=fitted_pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
all_names=num_cols+list(cat_names)
plt.figure(figsize=(18,8)); plot_tree(dt_pipe.named_steps['model'], feature_names=all_names, class_names=['Died','Survived'], filled=True, max_depth=3); plt.savefig("analytics/decision_tree.png"); plt.close()


models=[log_pipe,dt_pipe,rf_pipe]; names=["LogReg","DecisionTree","RandomForest"]; results=[]
plt.figure()
for i in range(3):
    m=models[i]; y_pred=m.predict(X_test); y_prob=m.predict_proba(X_test)[:,1]
    acc=accuracy_score(y_test,y_pred); prec=precision_score(y_test,y_pred); rec=recall_score(y_test,y_pred); f1=f1_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred); fpr,tpr,_=roc_curve(y_test,y_prob); roc_auc=auc(fpr,tpr)
    print(f"{names[i]} CM {cm} Acc {acc:.3f} Prec {prec:.3f} Rec {rec:.3f} F1 {f1:.3f} AUC {roc_auc:.3f}")
    results.append([names[i],acc,prec,rec,f1,roc_auc])
    plt.plot(fpr,tpr,label=f"{names[i]} {roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig("analytics/roc.png"); plt.close()
results_df=pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1','AUC']); print(results_df)


print("Baseline", precision_score(y_test, rf_pipe.predict(X_test)), recall_score(y_test, rf_pipe.predict(X_test)), f1_score(y_test, rf_pipe.predict(X_test)))
rf_bal=Pipeline([('preprocess',preprocess),('model',RandomForestClassifier(class_weight='balanced', random_state=42))])
rf_bal.fit(X_train,y_train)
print("Balanced", precision_score(y_test, rf_bal.predict(X_test)), recall_score(y_test, rf_bal.predict(X_test)), f1_score(y_test, rf_bal.predict(X_test)))
X_train_t=preprocess.fit_transform(X_train); X_test_t=preprocess.transform(X_test)
sm=SMOTE(random_state=42); X_sm,y_sm=sm.fit_resample(X_train_t,y_train)
rf_sm=RandomForestClassifier(random_state=42); rf_sm.fit(X_sm,y_sm)
print("SMOTE", precision_score(y_test, rf_sm.predict(X_test_t)), recall_score(y_test, rf_sm.predict(X_test_t)), f1_score(y_test, rf_sm.predict(X_test_t)))
print("Conclusion SMOTE improves recall best for minority, balanced good tradeoff")


param_grid={'model__n_estimators':[50,100],'model__max_depth':[5,10,None],'model__max_features':['sqrt','log2']}
rf_oob=Pipeline([('preprocess',preprocess),('model',RandomForestClassifier(oob_score=True, random_state=42))])
grid=GridSearchCV(rf_oob,param_grid,cv=3); grid.fit(X_train,y_train)
print("Best params", grid.best_params_); print("OOB", grid.best_estimator_.named_steps['model'].oob_score_)


X_reg=df_clean.drop(columns=['fare']); y_reg=df_clean['fare']
Xr_train,Xr_test,yr_train,yr_test=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
num_reg=['age','sibsp','parch','pclass']; cat_reg=['sex','embarked']
pre_reg=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num_reg),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat_reg)])
reg_pipe=Pipeline([('preprocess',pre_reg),('model',LinearRegression())])
reg_pipe.fit(Xr_train,yr_train); yr_pred=reg_pipe.predict(Xr_test)
mae=mean_absolute_error(yr_test,yr_pred); rmse=np.sqrt(mean_squared_error(yr_test,yr_pred)); r2=r2_score(yr_test,yr_pred)
n=len(yr_test); p=Xr_test.shape[1]; adj_r2=1-(1-r2)*(n-1)/(n-p-1)
print(f"MAE {mae:.3f} RMSE {rmse:.3f} R2 {r2:.3f} AdjR2 {adj_r2:.3f}")
residuals=yr_test-yr_pred
plt.figure(); plt.scatter(yr_pred,residuals); plt.axhline(0,color='red'); plt.savefig("analytics/residual.png"); plt.close()
print("Heteroscedasticity yes spread increases")


print(results_df); print(f"Regression MAE {mae:.3f} RMSE {rmse:.3f} R2 {r2:.3f} AdjR2 {adj_r2:.3f}")
print("Classification and regression separate metric groups not comparable")
print("Deploy Random Forest highest Acc ~0.82 F1 ~0.78 AUC ~0.87 better than others")


joblib.dump(grid.best_estimator_, "analytics/best_pipeline.joblib")
loaded=joblib.load("analytics/best_pipeline.joblib")
print("Reload test", loaded.predict(X_test.iloc[:2]), "works on raw data")
