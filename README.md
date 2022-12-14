# Система определения схожести названий

## Disclaimer

Это одна из первых попыток переноса кода обучения модели в формат скрипта, поэтому при возникновении ошибок, пожалуйста, воспользуйтесь нашим [colab-ноутбуком](https://colab.research.google.com/drive/1QeJKFc4gfYHwqdgpjYobJTcgQiZ7eO1g?usp=sharing)!

## Overview

Данный репозиторий содержит код обучения и разработки сервиса для определения схожести названий и брендов.

## Dataset

Для обучения модели использовался предоставленный заказчиками (менторами) датасет.

Он содежит 497819 пары дублирующих и не дублирующих названий.
При анализе датасета выяснилось, что доля дублирующих пар в датасете - приблизительно 0.07. 
Для сокращения затрат на обучение часть недублирующих пар была исключена из датасета так, чтобы процент дублирующих пар в датасете составлял 50%. Итого, использовалась выборка размерностью 7200 пар.

Размерность обучающей и тестовой выборок соотностятся как 80/20.

Размерность обучающей и валидационной выборок соотностятся как 80/20.

## Project structure

Проект имеет следующую структуру:
* model - папка со скриптами
  * utils - папка со служебными скриптами
	* prepare_data.py - разделение датасета на 3 выборки
	* evaluator.py - функции оценки качества модели 
  * trainer.py - скрипт обучения
  * pipeline.py - скрипт для использования модели
* conf.yml - конфигурационный файл
* requirements.txt - необходимые зависимости
* train.csv - файл с датасетом

Файл с датасетом можно взять [здесь](https://drive.google.com/file/d/1_YA5LcrHov--aHn16u99Aut8_sgqu4YX/view?usp=sharing).

## Training

При обучении модели использовался фреймворк [sentence-transformers](https://www.sbert.net/).
В качестве исходной использовалась модель `sentence-transformers/msmarco-distilbert-base-tas-b`. 
Для запуска обучения модели следует запустить скрипт 
```
cd model
python trainer.py
```

## Результаты
Для проверки работы системы запустите скрипт 
```
cd model
python pipeline.py количество_слов_в_базе имена_из_базы новые_имена
```
Например: `python pipeline.py 6 Ifmo
ITMO
MIRO
Institute
SSAU
Samara_University
ITMO_University
`

<p align="middle">
  <img src="example/example.png" width=350 />
</p>

Метрики модели:
* EmbeddingSimilarityEvaluator: (val data) 0.8635356105181481
* MSE: (val_data) 0.015562033612658599
* MSE: (test_data) 0.020477005394352963

## Эксперименты

Было проведено несколько экспериментов с разными параметрами, результаты представлены в таблице ниже:


|Loss function   |Количество эпох (epochs)   |Размер пакета (batch size)   |Частота проведения оценки (evaluation steps)   |EmbeddingSimilarityEvaluator (val data) |MSE (val_data) |MSE: (test_data) |
|---|---|---|---|---|---|---|
|CosineSimilarityLoss   |200   |32   |1000   |0.8551 |0.0437 |0.0501 |
|CosineSimilarityLoss   |400   |32   |1000   |0.8490 |0.0639 |0.0665 |
|CosineSimilarityLoss   |20   |32   |1000   |0.8631 |0.0165 |0.0205 |
|ContractiveLoss   |20   |32   |1000   |0.8631 |0.0831 |0.0784 |
|CosineSimilarityLoss   |20   |32   |146   |0.8635 |0.0156 |0.0205 |
|ContractiveLoss   |20   |32   |146   |0.8626 |0.0768 |0.0735 |
|CosineSimilarityLoss   |20   |16   |146   |0.8604 |0.0163 |0.0384 |

Как видно, лучшие значения метрик достригаются при следующих параметрах:
* Функция потерь - CosineSimilarityLoss
* 20 эпох
* Размер батча - 32
* Частота оценки качества - 146 шагов (в конце каждой эпохи)
