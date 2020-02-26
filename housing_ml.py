import os
import tarfile
import urllib
DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH=os.path.join("datasets","housing")
HOUSING_URL=DOWNLOAD_ROOT+"datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
housing=load_housing_data()
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()
import numpy as np
from zlib import crc32
def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff <test_ratio* 2**32
def split_train_set_by_id(data,test_ratio,id_coloumn):
    ids=data[id_coloumn]
    in_test_set=ids.apply(lambda id_:test_set_check(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]
def split_train_set(data,test_ratio):
    np.random.seed(42)
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)
housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
imputer=SimpleImputer(strategy="median")
housing_num=housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
X=imputer.transform(housing_num)
ordinalencoder=OrdinalEncoder()
housing_cat=housing[["ocean_proximity"]]
sa1=housing[["ocean_proximity"]]
sa=housing["ocean_proximity"]
housing_cat_encoded=ordinalencoder.fit_transform(housing_cat)
from sklearn.compose import ColoumnTransformer
num_attributes=list(housing_num)
cat_attributes=["ocean_proximity"]
from sklearn.pipeline import Pipeline
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy="median")),])
full_pipeline=ColoumnTransformer([("num",num_pipeline,num_attributes),("cat",OneHotEncoder(),cat_attributes)])