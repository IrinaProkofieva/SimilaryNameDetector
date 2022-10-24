import pandas as pd
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split


# Подготовка данных - разделение на три выборки
def prepare(df):
    # Сокращаем размерность датасета так, чтобы 0 и 1 было поровну
    df = pd.concat([df[df['is_duplicate'] == 0][:3600], df[df['is_duplicate'] == 1]])
    X = df[['name_1', 'name_2']]
    y = df[['is_duplicate']]
    # Выделяем тестовую часть
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Делим на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Переводим в формат, необходимый для модели
    train_data = [InputExample(texts=[x['name_1'], x['name_2']], label=float(y_train.loc[ndx]['is_duplicate'])) for
                  ndx, x in X_train.iterrows()]
    val_data = [InputExample(texts=[x['name_1'], x['name_2']], label=float(y_val.loc[ndx]['is_duplicate'])) for ndx, x
                in X_val.iterrows()]
    return train_data, val_data, [X_val, y_val, X_test, y_test]
