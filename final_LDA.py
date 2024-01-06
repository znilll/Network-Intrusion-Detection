import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report as clf_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path):
    col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]

    data = pd.read_csv(file_path, header=None, names=col_names)
    data.drop(['difficulty_level'], axis=1, inplace=True)

    return data

def change_label(df):
    df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable',
                      'smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
    df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named',
                      'phf','sendmail','snmpgetattack','snmpguess','spy','warezclient',
                      'warezmaster','xlock','xsnoop'],'R2L',inplace=True)
    df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
    df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack',
                      'xterm'],'U2R',inplace=True)

def normalize_data(train_data, test_data):
    # Combine training and test data
    combined_data = pd.concat([train_data, test_data], axis=0)

    numeric_col = combined_data.select_dtypes(include='number').columns
    std_scaler = StandardScaler()

    for i in numeric_col:
        arr = combined_data[i]
        arr = np.array(arr)
        combined_data.loc[:, i] = std_scaler.fit_transform(arr.reshape(len(arr), 1))

    cat_col = ['protocol_type', 'service', 'flag']
    categorical = combined_data[cat_col]
    categorical = pd.get_dummies(categorical, columns=cat_col)

    # creating a dataframe with multi-class labels (Dos, Probe, R2L, U2R, normal)
    multi_data = combined_data.copy()
    multi_label = pd.DataFrame(multi_data.label)

    # label encoding (0, 1, 2, 3, 4) multi-class labels (Dos, normal, Probe, R2L, U2R)
    le = preprocessing.LabelEncoder()
    enc_label = multi_label.apply(le.fit_transform)
    multi_data['intrusion'] = enc_label

    # one-hot-encoding attack label
    multi_data = pd.get_dummies(multi_data, columns=['label'], prefix="", prefix_sep="")
    multi_data['label'] = multi_label

    # creating a dataframe with only numeric attributes of multi-class dataset and encoded label attribute
    numeric_multi = multi_data[numeric_col].copy()
    numeric_multi.loc[:, 'intrusion'] = multi_data['intrusion']

    # joining the selected attribute with the one-hot-encoded categorical dataframe
    numeric_multi = numeric_multi.join(categorical)

    # then joining encoded, one-hot-encoded, and original attack label attribute
    multi_data = numeric_multi.join(
        multi_data[['intrusion', 'Dos', 'Probe', 'R2L', 'U2R', 'normal', 'label']], rsuffix='_suffix')

    train_data = multi_data[:len(train_data)]
    test_data = multi_data[len(train_data):]

    return train_data, test_data



def split_data(data):
    X = data.iloc[:, :-1].values
    y = data['intrusion'].values

    return X, y

def train_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

def train_lda(X, y):
    lda = LinearDiscriminantAnalysis(n_components = 2)
    X_lda = lda.fit(X, y).transform(X)
    return lda, X_lda


def train_knn(X, y, n_neighbors = 355):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn

def predict(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    target_names = ['Dos', 'Probe', 'R2L', 'U2R', 'Normal']
    classification_report_result = clf_report(y_test, y_pred, target_names=target_names, zero_division=1)
    return accuracy, classification_report_result


def visualize_pca(X_pca, y, save_path):
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF'])
    classes = np.unique(y)
    class_names = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']
    
    for cls, cls_name in zip(classes, class_names):
        plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], c=[cmap(cls)], label=cls_name, edgecolors='k')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Scatter Plot')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    
    
def visualize_lda(model, X, y, save_path=None):
    lda_transformed = model.transform(X)
    X_lda_1 = lda_transformed[:, 0]
    X_lda_2 = lda_transformed[:, 1]

    # Create a color map for the plot
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF'])
    classes = np.unique(y)
    class_names = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']

    plt.figure(figsize=(10, 8))
    for cls, cls_name in zip(classes, class_names):
        plt.scatter(X_lda_1[y == cls], X_lda_2[y == cls], c=[cmap(cls)], label=cls_name, edgecolors='k')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA Scatter Plot')
    plt.legend()

    # Save the plot to disk if save_path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def visualize_knn(X, y, save_path=None):
    X_knn_1 = X[:, 0]
    X_knn_2 = X[:, 1]

    # Create a color map for the plot
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF'])
    classes = np.unique(y)
    class_names = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']

    # Plot the scatter plot
    plt.figure(figsize=(10, 8))
    for cls, cls_name in zip(classes, class_names):
        plt.scatter(X_knn_1[y == cls], X_knn_2[y == cls], c=[cmap(cls)], label=cls_name, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Scatter Plot')
    plt.legend()

    # Save the plot to disk if save_path is provided
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

    
def main():
    train_data = load_data("KDDTrain.txt")
    test_data = load_data("KDDTest.txt")
    change_label(train_data)
    change_label(test_data)
    train_data, test_data = normalize_data(train_data, test_data)
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    x_pca = train_pca(X_train, 2)
    visualize_pca(x_pca, y_train, "Data_report/PCA_scatter.jpg")
    
    model_lda, X_lda = train_lda(X_train, y_train)
    y_pred = predict(model_lda, X_test)
    visualize_lda(model_lda, X_test, y_pred, save_path= "./Data_report/LDA_scatter.jpg")
    
    model_knn = train_knn(X_lda, y_train)
    X_test_lda = model_lda.transform(X_test)
    y_pred_knn = predict(model_knn, X_test_lda)
    visualize_knn(X_test_lda, y_pred_knn, save_path= "./Data_report/KNN_scatter.jpg")
    
    # model_knn_raw = train_knn(X_train, y_train)
    # y_pred_knn_raw = predict(model_knn_raw, X_test)


    accuracy_lda, classification_report_lda = evaluate_model(y_test, y_pred)
    print('LDA Accuracy: {}'.format(accuracy_lda))
    print(classification_report_lda)
    
    accuracy_knn, classification_report_knn = evaluate_model(y_test, y_pred_knn)
    print('KNN Accuracy: {}'.format(accuracy_knn))
    print(classification_report_knn)

    # accuracy_knn_raw, classification_report_knn_raw = evaluate_model(y_test, y_pred_knn_raw)      Raw data KNN
    # print('KNN Accuracy (Raw): {}'.format(accuracy_knn_raw))
    # print(classification_report_knn_raw)


    with open(r'Data_report\accuracy.txt', 'w') as file:
        file.write('Accuracy of LDA: {}\n'.format(accuracy_lda))
        file.write('\nClassification Report:\n')
        file.write(classification_report_lda)
        file.write('\n\nAccuracy of KNN: {}\n'.format(accuracy_knn))
        file.write('\nClassification Report:\n')
        file.write(classification_report_knn)
        # file.write('\n\nAccuracy of KNN (Raw): {}\n'.format(accuracy_knn_raw))        Raw data KNN
        # file.write('\nClassification Report (Raw):\n')
        # file.write(classification_report_knn_raw)

if __name__ == '__main__':
    main()
