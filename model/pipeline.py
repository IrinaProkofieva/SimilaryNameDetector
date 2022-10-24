from sentence_transformers import SentenceTransformer, util
import torch
import sys
import yaml


# Функия использования обученной модели. В параметры передаем модель, имена, которые уже есть в базе, и имена,
# которые будем проверять на дубликат
def pipeline(model, names_in_base, new_names):
    # Составляем эмбеддинги имен из базы
    e1 = model.encode(names_in_base, convert_to_tensor=True)
    # Составляем эмбеддинги новых имен
    e2 = model.encode(new_names, convert_to_tensor=True)
    # Строим матрицу косинусных расстояний
    cos_scores = util.cos_sim(e2, e1)
    # Для каждого имени выводим похожие имена из базы
    for i in range(len(new_names)):
        print(new_names[i])
        print('Возможно, вы имели в виду: ')
        # Похожими считаем те имена, расстояние до которых > 0.5
        similar = torch.nonzero(cos_scores[i] > 0.5).squeeze().tolist()
        if isinstance(similar, int):
            similar = [similar]
        for x in similar:
            print(names_in_base[x])


# Читаем параметры из конфигурационного файла
with open("../conf.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    model_name = cfg["pipeline"]["name"]

# Имена передаются в качестве параметров запуска программы: сначало число имен в базе, затем имена из базы,
# после чего новые имена
arr_len = int(sys.argv[1])
existed_names = sys.argv[2:2 + arr_len]
new_names = sys.argv[2 + arr_len:]
# Загружаем модель
model = SentenceTransformer(model_name)
# Запускаем проверку
pipeline(model, existed_names, new_names)
