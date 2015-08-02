import numpy as np
from selenium import webdriver
import requests, re
from gensim.models import Word2Vec
import pandas as pd

#gnews_model = Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)


def get_page_source(url, driver):
	"""
	Use the given webdriver to retrieve and return the source of the webpage at the given URL.
	If that fails, retrieve using requests.
	If *that* fails, return the empty string.

	Params:
		url: URL of desired webpage
		driver: webdriver (e.g. selenium.webdriver.PhantomJS())
	"""

	try:
		driver.get(url)
		return driver.page_source
	except:
		# use requests as a fallback -- HTTP only, no JS support
		try:
			return requests.get(url).text
		except:
			return ''

def vectorize_from_source(source_string, model, stopwords):
	"""
	Extract from source_string all substrings appearing in the given Word2Vec model's vocabulary and
	not appearing in the given list of stopwords. Vectorize those substrings according to the model and return the sum.

	Params:
		source_string: HTML source string
		model: Word2Vec model
		stopwords: list of stopwords
	"""
	words = [w for w in re.findall(r'[a-z]+',source_string.lower()) if w not in stopwords and w in model]
	if len(words)==0:
		return np.nan
	return sum((model[w] for w in words))

def vectorize_webpage(url, driver, model, stopwords):
	"""
	Obtain the source of the webpage with given URL. Extract from it all substrings
	appearing in the given Word2Vec model's vocabulary and not appearing in the given
	list of stopwords. Vectorize those substrings according to the model and return the sum.

	Params:
		url: URL of desired webpage
		model: Word2Vec model
		stopwords: list of stopwords
	"""

	source = get_page_source(url,driver)
	if source == '':
		#raise Exception('Error retrieving from URL:',url)
		return np.nan
	return vectorize_from_source(source, driver, model, stopwords)

def first(df):
	"""
	Return the first row of the DataFrame df. Used for aggregation.



	Params:
		df: DataFrame
	"""
    	return df.iloc[0]


def aggregate_ratings(df):
	"""
	Take a DataFrame of CrowdFlower ratings, formatted in the same way as actiondataset1.csv, and aggregate ratings
	by unit_id (equivalently, by site) by calculating the percentage of users that gave a "Yes" rating.

	Return the result of this aggregation, recording also the number of ratings included in each such calculation as rating_count.

	Params:
		df: DataFrame of CrowdFlower ratings, formatted as actiondataset1.csv
	"""

	agg_df = df.groupby('_unit_id').aggregate({'actionrating':lambda c: np.mean(c.apply(lambda s: 1 if s == 'Yes' else 0)),
												'action':first, 'action_inst':first, 'domain':first, 'site':first})
	agg_df['rating_count'] = df.groupby('_unit_id').actionrating.count()
	return agg_df

def get_source_series(url_series, driver = None):
	"""
	Given a Series of URLs of webpages, use either a webdriver or requests to retrieve the source of those
	webpages and return the resulting Series of strings.
	Params:
		url_series: pandas Series of URLs
		driver: Either None or a webdriver, e.g. an instance of selenium.webdriver.PhantomJS. If None, use requests instead.
	"""
	if driver is not None:
		return url_series.apply(lambda url:get_page_source(url,driver))
	else:
		return url_series.apply(lambda url:requests.get(url).text)

def vector_series_to_df(vector_ser):
	null_index = vector_ser[vector_ser.isnull()]
	not_null_index = vector_ser.index.sym_diff(null_index)

	X = vector_ser.ix[not_null_index].values
	X = np.hstack(X).reshape(len(X), len(X[0]))
	X = pd.DataFrame(X, index=not_null_index)

	for i in null_index:
		X.loc[i] = np.nan

	return X


def vector_df(agg_df, model, stopwords, retry=False):
	"""
	Return a DataFrame, indexed by _unit_id, whose ith row is the feature vector associated to agg_df.site[i]
	by the given Word2Vec model.

	Params:
		agg_df: DataFrame of aggregated ratings as returned by aggregate_ratings()
		model: Word2Vec model
		stopwords: list of stopwords
	"""
	source_ser = get_source_series(agg_df)
	vector_ser = source_ser.apply(lambda source: vectorize_from_source(source, model, stopwords))

	if retry:
		# If some sites failed to vectorize, retry them
		failed_index = vector_ser[vector_ser.isnull()].index
		while True:
			for i in failed_index:
			    vector_ser.ix[i] = vectorize_webpage(agg_df.site.ix[i], driver, model, stopwords=stopwords)
			failed_index = agg_df[vector_ser.isnull()].index
			if len(failed_index) == 0:
			    break

	if all(vector_ser.notnull()):
		X = vector_ser.values
		X = np.hstack(X).reshape(len(X),len(X[0]))
		return pd.DataFrame(X, index=vector_ser.index), True

	else:
		return vector_ser, False