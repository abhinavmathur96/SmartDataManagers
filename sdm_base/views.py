from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from .models import Child, Attribute

import pandas as pd
from django_pandas.io import read_frame
from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import graphviz
import base64
import numpy as np
import seaborn as sns
from io import BytesIO
from collections import defaultdict

# Cache

_ATTRIBUTES = ['friendliness', 'sportiness', 'kindness', 'talkativeness', 'extroversion', 'popularity',
               'gender', 'hardworking', 'intelligence', 'punctuality', 'creativity', 'leadership', 'disciplined', 'loyalty']

_FRIENDS = np.array(
    [
        [5, 3, 5, 4, 4, 4, 5, 5, 3, 4, 5, 4, 5], # Komal Marwah
        [5, 4, 3, 2, 1, 2, 4, 4, 3, 2, 4, 3, 5], # Aishwarya Gupta
        [4, 2, 3, 2, 3, 3, 4, 3, 3, 2, 4, 2, 5] # Abhinav Mathur
    ]
)

_DF_ATTR_STORE = {}
_DF_CORR_STORE = {}
_TOP_3_STORE = {}
_FRESH = defaultdict(bool)

_TREE_MODEL = {}
_RF_MODEL = {}
_KNN_MODEL = {}
_SVM_MODEL = {}

# Views


def index(request):
    return render(request, 'sdm_base/data_collection.html')


def dataVisualisation(request):
    if request.method == 'POST':
        try:
            rec = Child.objects.get(roll__iexact=request.POST.get('roll_no'))
            attr = read_frame(Attribute.objects.all().filter(
                child=rec.roll)).values[0, 1:].tolist()
            attr_list = []
            for a in _ATTRIBUTES:
                attr_list.append(int(str(request.POST.get(a+'_name'))))
            if attr_list == attr:
                return render(request, 'sdm_base/data_visualisation.html', {'name': rec.name, 'uid': rec.roll})
            return render(request, 'sdm_base/data_collection.html', {'message': 'User already exists'})
        except Child.DoesNotExist:
            newChild = Child(name=request.POST.get(
                'name', 'No Name'), roll=request.POST.get('roll_no'))
            newChild.save()

            newAttribute = Attribute(
                child=newChild,
                friendliness=request.POST.get('friendliness_name', 3),
                sportiness=request.POST.get('sportiness_name', 3),
                kindness=request.POST.get('kindness_name', 3),
                talkativeness=request.POST.get('talkativeness_name', 3),
                extroversion=request.POST.get('extroversion_name', 3),
                popularity=request.POST.get('popularity_name', 3),
                gender=request.POST.get('gender_name', 0),
                hardworking=request.POST.get('hardworking_name', 3),
                intelligence=request.POST.get('intelligence_name', 3),
                punctuality=request.POST.get('punctuality_name', 3),
                creativity=request.POST.get('creativity_name', 3),
                leadership=request.POST.get('leadership_name', 3),
                discipline=request.POST.get('disciplined_name', 3),
                loyalty=request.POST.get('loyalty_name', 3)
            )
            newAttribute.save()

            return render(request, 'sdm_base/data_visualisation.html', {'name': request.POST.get('name', 'No Name'), 'uid': newChild.roll})

    return render(request, 'sdm_base/data_visualisation.html')

# API calls


def fetchChildren(request):
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no')
        if my_roll:
            existing_friends = list(
                map(lambda x: x.roll, Child.objects.get(roll=my_roll).friends.all()))
            friend_count = len(existing_friends)
        res = Child.objects.defer('timestamp', 'friends')
        res = res.exclude(roll__iexact=request.POST.get(
            'roll_no')).values_list('roll', 'name')
        if friend_count > 0:
            return JsonResponse({'type': 'success', 'data': list(res), 'existing_friends': existing_friends, 'friend_count': friend_count}, safe=False)
        return JsonResponse({'type': 'success', 'data': list(res), 'friend_count': 0}, safe=False)
    else:
        return JsonResponse({'type': 'error', 'message': 'No records present'})


def addFriends(request):
    global _FRESH
    if request.method == 'POST':
        arr = request.POST.getlist('elements[]')
        my_roll = request.POST.get('roll_no', '')

        if len(arr) == 0 or my_roll == '':
            return JsonResponse({'type': 'error', 'message': 'No new friends to add. Check sent data'})

        try:
            my_friends = Child.objects.get(roll__exact=my_roll).friends
            count = 0

            for friends_roll in arr:
                if not my_friends.filter(roll=friends_roll).exists():
                    friend = Child.objects.get(roll__exact=friends_roll)
                    my_friends.add(friend)
                    count += 1

            if count > 0:
                _FRESH[my_roll] = False
            else:
                _FRESH[my_roll] = True

            return JsonResponse({'type': 'success', 'message': 'Added {} new friends'.format(count), 'count': count})
        except:
            return JsonResponse({'type': 'error', 'message': 'Some error occured while adding friends'})


def getCorrelationMatrix(request):
    global _TOP_3_STORE
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')

        df_attr = getCorrelationDataDF(my_roll)

        corr_matrix = df_attr.corr()
        top_3 = corr_matrix['friend']
        top_3 = list(zip(df_attr.columns.tolist(), top_3))
        top_3.sort(key=lambda x: x[1], reverse=True)
        _TOP_3_STORE[my_roll] = top_3

        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        sns_heatmap = sns.heatmap(
            corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
            cmap=sns.color_palette("RdBu_r", 11), mask=mask
        )
        figure = sns_heatmap.get_figure()
        data = getImageString(figure)

        return HttpResponse(data, content_type='image/png')


def trainDecisionTree(request):
    global _TREE_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        train, test = getAttrMatrix(my_roll, True)
        X_train = train.drop(labels=['friend', 'gender'], axis=1)
        Y_train = train['friend']
        X_test = test.drop(labels=['friend', 'gender'], axis=1)
        Y_test = test['friend']

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train.values, Y_train.values)
        # _TREE_MODEL[my_roll] = clf

        score = clf.score(X_test.values, Y_test.values)

        preds_tree = clf.predict(_FRIENDS)

        dot_data = tree.export_graphviz(clf, out_file=None, filled=True, feature_names=X_train.columns.tolist(),
                                        class_names=['Not friend', 'Friend'], rounded=True)
        return JsonResponse({'data': dot_data, 'score': score, 'pred': preds_tree.tolist()})


def predDecisionTree(request):
    global _TREE_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        preds = _TREE_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
        res = np.argmax(preds[0])
        return JsonResponse({'class': res, 'prob': preds[0, res]})


def trainSVM(request):
    global _SVM_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        train, test = getAttrMatrix(my_roll, True)
        X_train = train.drop(labels=['friend', 'gender'], axis=1)
        Y_train = train['friend']
        X_test = test.drop(labels=['friend', 'gender'], axis=1)
        Y_test = test['friend']

        clf = SGDClassifier(loss='log')
        clf.fit(X_train.values, Y_train.values)
        # _SVM_MODEL[my_roll] = clf

        preds_svm = clf.predict(_FRIENDS)

        score = clf.score(X_test.values, Y_test.values)

        return JsonResponse(
            {
                'type': 'success',
                'score': score if type(score) == float else score.tolist(),
                'coeff': clf.coef_.tolist(),
                'intercept': clf.intercept_.tolist(),
                'pred': preds_svm.tolist()
            }, safe=False
        )


def predSVM(request):
    global _SVM_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        preds = _SVM_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
        res = np.argmax(preds[0])
        return JsonResponse({'class': res, 'prob': preds[0, res]})


def trainRF(request):
    global _RF_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        train, test = getAttrMatrix(my_roll, True)
        X_train = train.drop(labels=['friend', 'gender'], axis=1)
        Y_train = train['friend']
        X_test = test.drop(labels=['friend', 'gender'], axis=1)
        Y_test = test['friend']

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train.values, Y_train.values)
        # _RF_MODEL[my_roll] = clf

        preds_rf = clf.predict(_FRIENDS)

        score = clf.score(X_test.values, Y_test.values)

        return JsonResponse({'score': score, 'pred': preds_rf.tolist()})


def predRF(request):
    global _RF_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        preds = _RF_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
        res = np.argmax(preds[0])
        return JsonResponse({'class': res, 'prob': preds[0, res]})


def trainKNN(request):
    global _KNN_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        train, test = getAttrMatrix(my_roll, True)
        X_train = train.drop(labels=['friend', 'gender'], axis=1)
        Y_train = train['friend']
        X_test = test.drop(labels=['friend', 'gender'], axis=1)
        Y_test = test['friend']

        clf = KNeighborsClassifier(n_neighbors=min(train.shape[0], 5))
        clf.fit(X_train.values, Y_train.values)
        # _KNN_MODEL[my_roll] = clf

        preds_knn = clf.predict(_FRIENDS)

        score = clf.score(X_test.values, Y_test.values)

        return JsonResponse({'score': score, 'pred': preds_knn.tolist()})


def predKNN(request):
    global _KNN_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        preds = _KNN_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
        res = np.argmax(preds[0])
        return JsonResponse({'class': res, 'prob': preds[0, res]})


def predict(request):
    global _TREE_MODEL, _SVM_MODEL, _RF_MODEL, _KNN_MODEL
    if request.method == 'POST':
        my_roll = request.POST.get('roll_no', '')
        if my_roll != '':
            if all([my_roll in _TREE_MODEL, my_roll in _SVM_MODEL, my_roll in _RF_MODEL, my_roll in _KNN_MODEL]):
                preds_tree = _TREE_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
                res_tree = np.argmax(preds_tree[0])
                preds_svm = _SVM_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
                res_svm = np.argmax(preds_svm[0])
                preds_rf = _RF_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
                res_rf = np.argmax(preds_rf[0])
                preds_knn = _KNN_MODEL[my_roll].predict_proba(_ABHINAV_MATHUR)
                res_knn = np.argmax(preds_knn[0])
                return JsonResponse(
                    {
                        'type': 'success',
                        'tree_pred': res_tree, 'tree_prob': preds_tree[0, res_tree],
                        'svm_pred': res_svm, 'svm_prob': preds_svm[0, res_svm],
                        'rf_pred': res_rf, 'rf_prob': preds_rf[0, res_rf],
                        'knn_pred': res_knn, 'knn_prob': preds_knn[0, res_knn]
                    }
                )
            return JsonResponse({'type': 'error', 'message': 'Keys not present in all', 'a':_TREE_MODEL.keys(), 'b':_SVM_MODEL.keys(), 'c':_RF_MODEL.keys(), 'd':_KNN_MODEL.keys()})
        return JsonResponse({'type': 'error', 'message': 'Something went wrong in prediction'})


# Helper Functions


def getAttrMatrix(my_roll, split=False):
    global _DF_ATTR_STORE, _FRESH
    attr_qs = Attribute.objects.all()
    friends_qs = Child.objects.get(roll__iexact=my_roll).friends.only('roll')

    df_attr = read_frame(attr_qs)
    df_friends = read_frame(friends_qs)['roll'].tolist()

    df_attr['friend'] = df_attr['child'].apply(
        lambda x: 1 if x in df_friends or x == my_roll else 0)
    df_attr.drop(labels=['child'], axis=1, inplace=True)
    df_attr.reset_index(drop=True, inplace=True)

    if split:
        # scikit_split = StratifiedShuffleSplit(
        #     n_splits=1, test_size=0.1, random_state=42)
        # for train_index, test_index in scikit_split.split(df_attr, df_attr['friend']):
        #     strat_train_set = df_attr.loc[train_index]
        #     strat_test_set = df_attr.loc[test_index]
        train_set, test_set = train_test_split(
            df_attr, test_size=0.1, random_state=42)
        return train_set, test_set
    else:
        _DF_ATTR_STORE[my_roll] = df_attr
        _FRESH[my_roll] = True
        return df_attr


def getImageString(figure, format='png'):
    figdata = BytesIO()
    figure.savefig(figdata, format=format, bbox_inches='tight')
    data = base64.b64encode(figdata.getvalue()).decode(
        'utf-8').replace('\n', '')
    figure.clf()
    figdata.close()
    return data


def getCorrelationDataDF(my_roll, split=False):
    global _DF_ATTR_STORE, _FRESH
    if split:
        return getAttrMatrix(my_roll, True)
    if my_roll not in _DF_ATTR_STORE or _FRESH[my_roll] != True:
        df_attr = getAttrMatrix(my_roll, split)
    else:
        df_attr = _DF_ATTR_STORE[my_roll]
    return df_attr
