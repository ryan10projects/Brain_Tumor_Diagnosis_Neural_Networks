from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults
import fastai
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate


def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Receive data from client
        Variance = float(request.POST.get('Variance'))
        Entropy = float(request.POST.get('Entropy'))
        Skewness = float(request.POST.get('Skewness'))
        Kurtosis = float(request.POST.get('Kurtosis'))
        ASM = float(request.POST.get('ASM'))
        Homogeneity = float(request.POST.get('Homogeneity'))
        Dissimilarity = float(request.POST.get('Dissimilarity'))
        Correlation = float(request.POST.get('Correlation'))

        # Unpickle model
        model = pd.read_pickle(r"new.pickle")
        # Make prediction
        result = model.predict([[Variance, Entropy, Skewness, Kurtosis, ASM, Homogeneity, Dissimilarity, Correlation]])

        classification = result[0]

        PredResults.objects.create(Variance=Variance, Entropy=Entropy, Skewness=Skewness,
                                   Kurtosis=Kurtosis,ASM=ASM ,Homogeneity=Homogeneity,Dissimilarity=Dissimilarity,Correlation=Correlation, classification=classification)

        return JsonResponse({'result': classification, 'Variance': Variance,
                             'Entropy': Entropy, 'Skewness': Skewness, 'Kurtosis': Kurtosis,'ASM':ASM,'Homogeneity':Homogeneity,'Dissimilarity':Dissimilarity,'Correlation':Correlation},
                            safe=False)


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request,"results.html",data)
