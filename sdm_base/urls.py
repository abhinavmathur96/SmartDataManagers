from django.urls import path

from . import views

urlpatterns = [
    path(r'dataviz/', views.dataVisualisation, name='dataviz'),
    path(r'api/fetch_children/', views.fetchChildren, name='fetchChildren'),
    path(r'api/add_friends/', views.addFriends, name='addFriends'),
    path(r'api/get_correlation_matrix/', views.getCorrelationMatrix, name='getCorrelationMatrix'),
    path(r'api/train_decision_tree/', views.trainDecisionTree, name='trainDecisionTree'),
    path(r'api/train_svm/', views.trainSVM, name='trainSVM'),
    path(r'api/train_rf/', views.trainRF, name='trainRF'),
    path(r'api/train_knn/', views.trainKNN, name='trainKNN'),
    path(r'api/predict/', views.predict, name='predict'),
    path(r'', views.index, name='index')
]