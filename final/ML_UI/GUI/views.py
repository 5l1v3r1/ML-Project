from django.shortcuts import render
from django.http import JsonResponse
from .lib.Library.FastTextModel import FastTextModel
from .lib.Library.Aahaber import Aahaber
from .lib.Library.TurkishProcessor import TurkishProcessor
FastTextModel

datasets = ['Aahaber', 'Milliyet', 'Hurriyet']
algorithms = ['MLP', 'RNN', 'CNN']


def home(request):
    context = {
        'datasets': datasets,
        'algorithms': algorithms
    }
    return render(request, 'gui/home.html', context)


def about(request):
    return render(request, 'gui/about.html', {'t': 'test'})


def evaluate(request):

    H = Aahaber(False, True)
    tp = TurkishProcessor(H)
    mm = FastTextModel(tp, H)
    # mm.evaluate()
    data = {
        'name': 'Vitor',
        'location': 'Finland',
        'is_active': True,
        'count': 28
    }
    return JsonResponse(data)
