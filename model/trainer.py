from torch.utils.data import DataLoader
import yaml
from sentence_transformers import losses, SentenceTransformer, evaluation
import pandas as pd
from utils.prepare_data import prepare
from utils.evaluator import print_evaluation


# Функция обучения
def train(model, epochs, batch_size, warmup_steps, train_data, train_loss, evaluator, evaluation_steps):
    print("TRAINING STARTED")
    # Загружаем тренировочную выборку
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # Запускаем обучение с заданными параметрами
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=epochs,
              evaluation_steps=evaluation_steps,
              warmup_steps=warmup_steps,
              evaluator=evaluator)
    print("TRAINING FINISHED")


# Берем параметры обучения из конфигурационного файла
with open("../conf.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    train_cfg = cfg["train"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    warmup_steps = train_cfg["warmup_steps"]
    evaluation_steps = train_cfg["evaluation_steps"]
    dataset_path = cfg["data"]["dataset_path"]
    model_name = cfg["model"]["name"]
    model_path = cfg["model"]["path"]

# Читаем данный нам датасет
df = pd.read_csv(dataset_path, delimiter=',', decimal='.', index_col='pair_id')
# Делим на тренировочную, валидационную и тестовую выборки
train_data, val_data, valid_sets = prepare(df)
X_val, y_val, X_test, y_test = valid_sets
# Загружаем предобученную модель
model = SentenceTransformer(model_name)
# Определяем функцию потерь
train_loss = losses.CosineSimilarityLoss(model)
# Определяем валидационную функцию
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_data, name='sts-dev')
# Запускаем обучение
train(model, epochs, batch_size, warmup_steps, train_data, train_loss, evaluator, evaluation_steps)
# Сохраняем модель
model.save(model_path)
# Выводим метрики
print_evaluation(model, X_test, y_test, X_val, y_val, evaluator)
