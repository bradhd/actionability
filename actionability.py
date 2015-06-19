from StringIO import StringIO
import pandas as pd
import requests, re, itertools, gensim, pickle, sys
from gensim import corpora, models, similarities
import numpy as np
from lxml import html, etree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def y_summary(y_test, y_pred):
    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = sum(y_pred == y_test)*1.0/len(y_test)
    true_proportion = sum(y_test)*1.0/len(y_test)
    predicted_proportion = sum(y_pred)*1.0/len(y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return precision, recall, accuracy, true_proportion, predicted_proportion, auc
    #print '--'
    #print 'precision: %f\nrecall: %f\naccuracy: %f\ntrue_proportion: %f\nauc: %f' % (precision, recall, accuracy, true_proportion, auc)

def strings_to_tokens_list(strings, multiplicity=False):
    """
    Given a list of strings, converts to lowercase and returns the list of all [a-z]+ substrings
    found among strings in that list, with or without multiplicity.
    """
    
    token_lists = [re.findall(r'[a-z]+',s.lower()) for s in strings]
    if not multiplicity:
        return list(np.unique(list(itertools.chain(*token_lists))))
    else:
        return list(itertools.chain(*token_lists))

def elements_to_text_tokens(elements, multiplicity=False):
    texts = [e.text.lower() for e in elements if e.text is not None]
    return strings_to_tokens_list(texts, multiplicity)

def elements_to_attr_value_tokens(elements, attr, multiplicity=False):
    """
    elements is a list of Element objects that have values for the attribute attr.
    """
    values = [e.get(attr) for e in elements]
    return strings_to_tokens_list(values, multiplicity)

def attr_pat_dict_to_xpath(attr_pat_dict):
    """Helper function for constructing XPath expressions."""
    if len(attr_pat_dict)==0:
        return ''
    return '['+ ' and '.join(["re:match(@%s, '%s')" % (attr,attr_pat_dict[attr]) for attr in attr_pat_dict.keys()])+']'

def root_to_tokens_list(root, attr, element_type=None, attr_pat_dict={}, multiplicity=False):
    """
    Find all Elements from root that have attributes matching the regexes defined
    by attr_pat_dict, and return the list of all maximal alphabetical substrings appearing among their 
    values of attr. Or if attr='text', do the same thing using their texts instead of an attribute value.
    """
    if element_type is None:
        element_type = '*'
    if attr != 'text' and attr not in attr_pat_dict.keys():
        attr_pat_dict[attr] = '.+'
    elements= root.xpath("//%s%s" % (element_type, attr_pat_dict_to_xpath(attr_pat_dict)), 
        namespaces={"re": "http://exslt.org/regular-expressions"})   
    if attr != 'text':
        return elements_to_attr_value_tokens(elements, attr, multiplicity=multiplicity)
    else:
        #print elements, root, attr, element_type
        return elements_to_text_tokens(elements, multiplicity=multiplicity)
    
def html_to_tokens_list(raw_html, attr, element_type=None, attr_pat_dict={}, multiplicity=False):
    parser = etree.HTMLParser(encoding='utf-8')
    root = etree.parse(StringIO(raw_html.encode('utf-8')), parser)
    
    return root_to_tokens_list(root, attr, element_type, attr_pat_dict, multiplicity)

def url_to_tokens_list(url, attr, element_type=None, attr_pat_dict={}, multiplicity=False):
    raw_html = requests.get(url).content
    return html_to_tokens_list(raw_html, attr, element_type, attr_pat_dict, multiplicity)

def root_to_link_text_token_string(root):
    return ' '.join(root_to_tokens_list(root, 
                                        attr='text', 
                                        attr_pat_dict={'href':'.+'}, 
                                        multiplicity=True))

def html_to_link_text_token_string(txt):
    parser = etree.HTMLParser(encoding='utf-8')
    root = etree.parse(StringIO(txt.encode('utf-8')), parser)
    return root_to_link_text_token_string(root)
    #return root_to_link_text_token_string(etree.HTML(txt.encode('utf-8')))

def root_to_button_name_token_string(root, multiplicity=False):
    # find tokens from text of elements with 'button' as a substring of the attribute 'class'
    tokens1 = root_to_tokens_list(root, attr='text', attr_pat_dict={'class':r'.*button'}, multiplicity=True)
    
    # find tokens from 'value' attributes of input-type elements with 'type' attribute 'button' or 'submit'
    tokens2 = root_to_tokens_list(root, attr='value', element_type='input', attr_pat_dict={'type':r'(?:button)|(?:submit)'}, multiplicity=True)
    
    # find tokens from text of button-type elements
    tokens3 = root_to_tokens_list(root, attr='text', element_type='button', multiplicity=True)
    
    if multiplicity:
        return ' '.join(tokens1 + tokens2 + tokens3)
    else:
        return ' '.join(np.unique(tokens1 + tokens2 + tokens3))

def html_to_button_name_token_string(txt):
    parser = etree.HTMLParser(encoding='utf-8')
    root = etree.parse(StringIO(txt.encode('utf-8')), parser)
    return root_to_button_name_token_string(root, multiplicity=True)

def url_to_clickable_token_string(url):
    txt = requests.get(url).content
    root = etree.HTML(txt)
    return root_to_button_name_token_string(root) + ' ' + root_to_link_text_token_string(root)

def cross_validate_lolo(X_token_strings, y, labels, clfs, use_tfidf=False):
    """
    Perform leave-one-label-out cross-validation, converting strings to bag-of-words vectors.
    Return a DataFrame summarizing the results.
    """
    all_y_test = np.array([])
    all_y_pred = np.array([])
    results = []
    for clf in clfs:
        X_token_lists = np.array([s.split(' ') for s in X_token_strings])
        labels = np.array(labels)

        for label in np.unique(labels):       
            #print label
            # set aside everything with given label as test set
            train_index = (labels != label)
            test_index = (labels == label)

            X_train_token_lists = X_token_lists[train_index]

            # construct gensim Dictionary from training set
            dictionary = corpora.Dictionary(X_train_token_lists)
            num_terms = len(dictionary.token2id)

            # convert all token lists to vectors
            X_gensim_corpus = map(dictionary.doc2bow, X_token_lists)

            if use_tfidf:
                train_corpus = map(dictionary.doc2bow, X_train_token_lists)
                tfidf = models.TfidfModel(train_corpus)
                X = gensim.matutils.corpus2csc(tfidf[X_gensim_corpus], num_terms=num_terms).T

            else:
                X = gensim.matutils.corpus2csc(X_gensim_corpus, num_terms=num_terms).T
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]    

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
            accuracy = sum(y_pred == y_test)*1.0/len(y_test)
            true_proportion = sum(y_test)*1.0/len(y_test)
            predicted_proportion = sum(y_pred)*1.0/len(y_pred)
            auc = roc_auc_score(y_test, y_pred)
            results += [(label, len(y_test), accuracy, precision, recall, true_proportion, predicted_proportion, auc)]
            all_y_test = np.append(all_y_test, y_test)
            all_y_pred = np.append(all_y_pred, y_pred)
    result_df = pd.DataFrame(results)
    result_df.columns = ['domain','page_count','accuracy','precision','recall', 'true_proportion', 'predicted_proportion', 'auc']
    return result_df, all_y_test, all_y_pred