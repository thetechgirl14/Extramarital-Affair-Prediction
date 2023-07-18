from data_preprocessing import load_data, preprocess_data
from data_visualization import visualize_affair_distribution, visualize_relationships
from data_processing import handle_imbalanced_data, scale_data
from training_model import train_model
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
import os

def create_figure_directory():
    directory = "figures"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Directory Created')

def main():
    create_figure_directory()

    dta = load_data()
    dta = preprocess_data(dta)
    visualize_affair_distribution(dta, 'affair_distribution')

    dta = handle_imbalanced_data(dta.drop('affair', axis=1), dta['affair'])
    visualize_affair_distribution(dta, 'resampled_affair_distribution')
    visualize_relationships(dta)

    X = dta.drop('affair', axis=1)
    y = dta['affair']

    X_scaled = scale_data(X)

    clf_LR, X_train, y_train, X_test, y_test = train_model(X_scaled, y)
    train_accuracy, test_accuracy, cm, report, fpr, tpr, roc_auc = evaluate_model(clf_LR, X_train, y_train, X_test, y_test)
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)

    print(f"Training set accuracy score: {train_accuracy}")
    print(f"Testing set accuracy score: {test_accuracy}")
    print(f"\nClassification Report:\n{report}")

if __name__ == '__main__':
    main()
