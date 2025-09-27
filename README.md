# Повышение устойчивости алгоритмов машинного обучения для сегментации мультиспектральных разновременных спутниковых снимков

## Полезные ссылки
Похожие работы:
* https://www.kaggle.com/code/nisaneretva/multiclass-segmentation-deepglobe-with-unet
* https://colab.research.google.com/drive/1FaawyFDpfF2eMn8jiLRXrY7Jjwh54nWB?usp=sharing
* https://github.com/iterative/cml

## Цель

## Описание проекта
### Функция потерь
При обучении модели используется функция потерь `CrossEntropyLoss` из библиотеки [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 
В документации сказано, что внутри самой функции применяется `softmax`, поэтому внутри модели нам не нужно этого делать, то есть модель должна возвращать ненормализованные логиты 

## Запуск