#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

##first attempt- w_o created feature
#features_list = ['poi', 'salary', 'total_payments', 'deferral_payments', 'exercised_stock_options', 'bonus',
#                 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'loan_advances', 'other',
#                 'from_this_person_to_poi', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

##second attempt- edit variables from/to poi to be ratios
#features_list = ['poi', 'salary', 'total_payments', 'deferral_payments', 'exercised_stock_options', 'bonus',
#                 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'loan_advances', 'other',
#                 'to_poi_ratio', 'deferred_income', 'long_term_incentive', 'from_poi_ratio']

##third attempt- removing non-imporant features
#features_list = ['poi', 'salary', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock',
#                 'other', 'to_poi_ratio', 'deferred_income']

##fourth attempt- removed salary based on importance- good performance with Dec Tree & AdaBoost
#features_list = ['poi', 'total_payments', 'exercised_stock_options', 'bonus',
#                 'other', 'to_poi_ratio']

##try lumping important finance features into one feature (addition)- worse performance when grouping into 1 feature
#features_list = ['poi', 'fin_addition', 'to_poi_ratio']

##try lumping important finance features into one feature (multiplication)- worse performance when grouping into
#1 feature
#features_list = ['poi', 'fin_multiplication', 'to_poi_ratio']

##fifth (and final) attempt- removed bonus based on importance- strong performance with Dec Tree & AdaBoost
features_list = ['poi', 'total_payments', 'exercised_stock_options', 'other', 'to_poi_ratio']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers

##remove the total record from dictionary
data_dict.pop('TOTAL', None)
##remove random entry from dictionary which clearly isn't a "'person' of interest"
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
##edit BELFER based on pdf/dataset inconsistencies- identified by odd negative values for certain features
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['deferral_income'] = -102500
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'

##edit BHATNAGAR based on pdf/dataset inconsistencies- identified by odd negative values for certain features
data_dict['BHATNAGAR SANJAY']['deferral_payments'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

##remove BHATNAGAR restricted stock deferred data point as an outlier
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = 'NaN'



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
min_fin = 10000
max_fin = 0
for individual in data_dict:
    sent_to_poi = data_dict[individual]['from_this_person_to_poi']
    rec_from_poi = data_dict[individual]['from_poi_to_this_person']
    to_messages = data_dict[individual]['to_messages']
    from_messages = data_dict[individual]['from_messages']
    tot_payments = data_dict[individual]['total_payments']
    excstockopt_payments = data_dict[individual]['exercised_stock_options']
    bonus = data_dict[individual]['bonus']
    other = data_dict[individual]['other']

    #creat ratio features for poi related email counts
    if sent_to_poi <> 'NaN' and  to_messages <> 'NaN':
        my_dataset[individual]['to_poi_ratio'] = 1.0 * sent_to_poi / to_messages
    else:
        my_dataset[individual]['to_poi_ratio'] = 'NaN'
    if rec_from_poi <> 'NaN' and  from_messages <> 'NaN':
        my_dataset[individual]['from_poi_ratio'] = 1.0 * rec_from_poi / from_messages
    else:
        my_dataset[individual]['from_poi_ratio'] = 'NaN'

    #creating addition/multiplication of fin features as 1 feature

    if tot_payments <> 'NaN' or excstockopt_payments <> 'NaN' or bonus <> 'NaN' or other <> 'NaN':
        if tot_payments == 'NaN':
            tot_payments_add = 0
            tot_payments_mult = 1
        else:
            tot_payments_add = tot_payments
            tot_payments_mult = tot_payments
        if excstockopt_payments == 'NaN':
            excstockopt_payments_add = 0
            excstockopt_payments_mult = 1
        else:
            excstockopt_payments_add = excstockopt_payments
            excstockopt_payments_mult = excstockopt_payments
        if bonus == 'NaN':
            bonus_add = 0
            bonus_mult = 1
        else:
            bonus_add = bonus
            bonus_mult = bonus
        if other == 'NaN':
            other_add = 0
            other_mult = 1
        else:
            other_add = other
            other_mult = other
        my_dataset[individual]['fin_addition'] = tot_payments_add + excstockopt_payments_add +other_add +bonus_add
        my_dataset[individual]['fin_multiplication'] = tot_payments_mult * excstockopt_payments_mult * other_mult\
                                                       * bonus_mult
    else:
        my_dataset[individual]['fin_addition'] = 'NaN'
        my_dataset[individual]['fin_multiplication'] = 'NaN'


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##first classfier attempted- great for precision, below average for recall (based on final features list)
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
#clf.fit(features,labels)


##second classfier- entropy is the optimal criterion. Strong precision and recall
#from sklearn import tree
#from sklearn.grid_search import GridSearchCV
#parameter = {'criterion':['entropy','gini'],
#               'min_samples_split':[2,4,10]}
#dec_tree = tree.DecisionTreeClassifier()
#clf = GridSearchCV(dec_tree,parameter)
#clf.fit(features,labels)
#print clf.best_estimator_


##third classifier great for precision, poor for recall (based on final features list)
#from sklearn import ensemble
#clf = ensemble.RandomForestClassifier()
#clf = clf.fit(features, labels)
#print clf.feature_importances_


##fourth classifier strong for precision and recall (based on final features list)- use findings from 2nd classifier
from sklearn import ensemble, tree
#from sklearn.grid_search import GridSearchCV
#parameter = {'algorithm':['SAMME', 'SAMME.R'],
#                'n_estimators':[2,5,10,25,50]}
#clf = GridSearchCV(ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion='entropy'),
# n_estimators=50),parameter)
#clf = clf.fit(features,labels)
#print clf.best_estimator_

##fifth classifier strong for precision and recall (based on final features list)- use findings from classifiers 2&4
from sklearn import ensemble, tree
tree = tree.DecisionTreeClassifier(criterion='entropy')
clf = ensemble.AdaBoostClassifier(base_estimator=tree, algorithm='SAMME',n_estimators=50)
clf = clf.fit(features,labels)
print clf.feature_importances_
print features_list[1:]

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


##creating new test classifier using kfold cross validation
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier_kfold(clf, dataset, feature_list, folds):
    from sklearn.cross_validation import StratifiedKFold
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedKFold(labels, n_folds=folds)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf

##running through different fold inputs of k-fold cross-validation
#folds = [2,3,5,10]
#for each in folds:
#    test_classifier_kfold(clf,my_dataset,features_list,each)


##test the algorithm multiple times and obtain the accuracy, precision, and recall averages
tot_accuracy = 0
tot_precision = 0
tot_recall = 0
i = 0
while i < 10:
    accuracy, precision, recall = test_classifier(clf, my_dataset, features_list)
    tot_accuracy += accuracy
    tot_precision += precision
    tot_recall += recall
    i += 1
print tot_accuracy/float(10), tot_precision/float(10), tot_recall/float(10)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)