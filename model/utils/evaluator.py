from sentence_transformers import util
from sklearn.metrics import mean_squared_error


# Вычисление MSE
def evaluate(model, X_test, y_test):
    # Вычисляем эмбеддинги для списков новых и старых имен
    embeddings1 = model.encode(X_test['name_1'].to_numpy(), convert_to_tensor=True)
    embeddings2 = model.encode(X_test['name_2'].to_numpy(), convert_to_tensor=True)
    y_t = y_test['is_duplicate'].to_numpy()

    # Вычисляем косинусную схожесть
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    y_pred = []
    for i in range(X_test['name_1'].shape[0]):
        y_pred.append(cosine_scores[i][i].item())

    return mean_squared_error(y_t, y_pred)


def print_evaluation(model, X_test, y_test, X_val, y_val, evaluator):
    print('EmbeddingSimilarityEvaluator: (val data)', model.evaluate(evaluator))
    print('MSE: (val_data)', evaluate(model, X_val, y_val))
    print('MSE: (test_data)', evaluate(model, X_test, y_test))
