from hypothesis.strategies import *
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


df = pd.DataFrame(
        {'city': ['London', 'London', 'Paris', 'Sallisaw'],
            'title': ["His Last Bow", "How Watson Learned the Trick",
                "A Moveable Feast", "The Grapes of Wrath"],
            'expert_rating': [5, 3, 4, 5],
            'user_rating': [4, 5, 4, 3]})


ct = ColumnTransformer(
    [('city_category', OneHotEncoder(dtype='int'), 'city')],
    remainder='drop')


ct.fit(df)
ct.get_feature_names()
ct.transform(df).toarray()
