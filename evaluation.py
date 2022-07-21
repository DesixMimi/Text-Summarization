from rouge import Rouge
import matplotlib.pyplot as plt
import seaborn as sn

def compute_rouge(predictions, targets):
    predictions = [" ".join(prediction).lower() for prediction in predictions]
    predictions = [prediction if prediction else "EMPTY" for prediction in predictions]
    targets = [" ".join(target).lower() for target in targets]
    targets = [target if target else "EMPTY" for target in targets]
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, refs=targets, avg=True)
    return scores['rouge-2']['f']


def plotConfusionMatrix(model, X_test, y_test):
    # Plot confusion matrix

    # Gets predicted labels from model
    y_pred = np.around(model.predict(X_test)).astype(int).flatten()

    # Generates confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Formats and displays the confusion matrix
    figure(num=None, figsize=(8, 6), dpi=300)
    df_cm = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Pred Negative', 'Pred Positive'])
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 13}, cmap=plt.cm.Blues, fmt='g', cbar=False)
    sn.set(rc={'figure.figsize': (5, 3)})
    plt.title('Binary Classification Confusion Matrix', fontsize=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.show()