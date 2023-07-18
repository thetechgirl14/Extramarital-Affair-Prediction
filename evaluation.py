from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt

def evaluate_model(clf_LR, X_train, y_train, X_test, y_test):
    y_train_pred = clf_LR.predict(X_train)
    y_pred = clf_LR.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    pred_probab = clf_LR.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred_probab[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = [0, 1]
    accuracy_values = [train_accuracy, test_accuracy]
    labels = ['Training Accuracy', 'Testing Accuracy']
    colors = ['#1f77a4', '#1f7f1e']

    ax.bar(x_pos, accuracy_values, align='center', color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Accuracy vs. Testing Accuracy', fontsize=14)

    for i, v in enumerate(accuracy_values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    
    plt.savefig('figures/accuracy_plot.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.text(0, 0, report, fontsize=12)
    ax.axis('off')
    plt.savefig('figures/classification_report.png', bbox_inches='tight')
    plt.close()

    return train_accuracy, test_accuracy, cm, report, fpr, tpr, roc_auc

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=20)
    plt.colorbar()
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])
    plt.grid(False)
    plt.savefig("figures/confusion_matrix.png")
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=20)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.savefig("figures/roc_curve.png")
    plt.close()
