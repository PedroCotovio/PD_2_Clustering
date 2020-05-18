import pandas as pd
import os
from yellowbrick.cluster import KElbowVisualizer
from silhouette import silhouette_visualizer
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import fowlkes_mallows_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def get_path(path='dataset'):

    """

    :param path: str, path to datasets folder from current path
    :return: path to datasets folder
    """
    
    pth = os.getcwd()
    pth = os.path.join(pth, str(path))
    print('Path to Data: {}'.format(pth))
    return pth


def validate_format(df, rows=110, columns=5147, target='DIAGNOSIS', drop=[], col_print=5):

    """
    Validate dataframe format, preprocess data and print important info

    :param df: dataframe
    :param rows: int, number of rows
    :param columns: int, number of columns
    :param target: str, target column name in dataframe
    :param drop: array of str, list of column names
    :param col_print: int, number of columns name to print
    :return: (dataframe, features matrix, target), processed data
    """
    
    df = pd.DataFrame(df)
    rows = int(rows)
    columns = int(columns)
    col_print = int(col_print)
    
    extra = ''
    
    if columns > col_print:
        extra = 'First {} '.format(col_print)

    try:
        y = df[str(target)]
        drop.append(str(target))
        df.drop(drop, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if df.shape == (rows, columns):
        
            X = df.values
            
            print('Valid Format')
            print('')
            print('{}Columns: {}'.format(extra, list(df.columns[:col_print])))
            print('')
            print('Targets: {}'.format(set(y)))
            
            df['target'] = y
            
            return df, X, y
    
        else:
            raise AttributeError('No Valid Format Found')
            
    except KeyError:
        print("Dataframe already processed or columns don't exist")


def variance_threshold(X, threshold=None):

    """
    Removes Features with low variance

    :param X: Array of Arrays, Features Matrix
    :param threshold: Variance threshold
    :return: Tranformed matrix
    """
    
    #Transform Data
    if threshold:
        selector = VarianceThreshold(threshold)
    else:
        selector = VarianceThreshold()
        
    X_new = selector.fit_transform(X)
    
    print('{} Features Kept of {}'.format(X_new.shape[1], X.shape[1]))
    
    return X_new


def detect_threshold(X, _range=15):

    """
    Plot feature dropout

    :param X: Array of Arrays, Features Matrix
    :param _range: int, number of variance points to check
    :return: tuple, (array, X lengths at '_range' variance threshold; array, variance thresholds)
    """
    
    var_range = np.linspace(0, 1, int(_range))

    lenght = X.shape[1]
    lenghts = []
    _vars = []
    for i in var_range:
        selector = VarianceThreshold(i)
        X_new = selector.fit_transform(X)
        
        if X_new.shape[1] < lenght:
            lenght = X_new.shape[1]
            lenghts.append(lenght)
            _vars.append(i)
            
    fig, ax = plt.subplots(figsize=(9,7))
    ax.plot(_vars, lenghts)
    ax.set_title('Features Dropout', fontsize=15, fontweight='demi')
    ax.set_ylabel('Number of Features')
    ax.set_xlabel('Variance Threshold')

    return lenghts, _vars


def ca_threshold(X, y=None, analisys='pca', threshold=0.8):

    """
    Transform Features Matrix by applying dimensionality reduction techniques, LDA or PCA

    :param X: Array of Arrays, Features Matrix
    :param y: Array, Target, only used for LDA analysis, else ignored
    :param analisys: str, Type of analysis ('pca' or 'lda')
    :param threshold: float, Explained variance threshold
    :return: array, Transformed Features Matrix
    """
    
    threshold = float(threshold)

    if 0 > float(threshold) or 1 < float(threshold):

        print('threshold parameter must be a float between 0 and 1')
        return None
    
    # standardize data
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    
    if analisys == 'pca':
        
        # get complete explained variance ratios
        
        model = PCA()
        model.fit(X_scale)
        var_explained = np.cumsum(model.explained_variance_ratio_)
        
        # reduce data
    
        model = PCA(n_components=threshold, svd_solver='full')
        X_model = model.fit_transform(X_scale)
        
        # plot results

        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(var_explained)
        ax.vlines(X_model.shape[1], var_explained[0], var_explained[-1])
        ax.set_title('Explained Variance', fontsize=15, fontweight='demi')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        
        # Calculate Loss
        
        X_projected = model.inverse_transform(X_model)
        loss = ((X - X_projected) ** 2).mean()

        print('Evaluation loss: ', round(loss, 4), ' (MSE)')

    else:
        # get complete explained variance ratios
        
        model = LinearDiscriminantAnalysis()
        model.fit(X_scale, y)
        
        # reduce data
        X_model = model.transform(X_scale)
        
    print("Components Kept explain {}% of the dataset's variance".format(threshold*100))
    print('{} Components Kept from {} features'.format(X_model.shape[1], X.shape[1]))
    
    return X_model


# Bar Chart
def make_bar_chart(x):
    """
    Return Bar Chart

    :param x: array, dataset to plot

    """
    plt.figure(figsize=(14, 6));
    g = sns.countplot(x)
    ax = g.axes
    for p in ax.patches:
        ax.annotate(f"{p.get_height() * 100 / x.shape[0]:.2f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
                    textcoords='offset points')
        
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def generate_representation(X, train=True, path='', names=['encoder', 'autoencoder'], encoding_dim=20, batch=None, epochs=100, split=0.05, verbose=0):
    
    X = np.array(X)
    names = list(names)
    
    encoder_path = os.path.join(names[0], str(path))
    autoencoder_path = os.path.join(names[1], str(path))
    
    if train is True:
        
        n_columns = X.shape[1]
        encoding_dim = int(encoding_dim)
        epochs = int(epochs)
        split = float(split)

        if not batch:
            batch = int(round(X.shape[0]/100, 0))

        input_df = Input(shape=(n_columns,))
        encoded = Dense(encoding_dim, activation='relu')(input_df)
        decoded = Dense(n_columns, activation='sigmoid')(encoded)

        # Full Model (decoder)
        autoencoder = Model(input_df, decoded)

        # Generates representations (encoder)
        encoder = Model(input_df, encoded)

        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        autoencoder.fit(X, X,
                        epochs=epochs,
                        batch_size=batch,
                        shuffle=True,
                        validation_split=split,
                        verbose=verbose)
        
        autoencoder.save(autoencoder_path)
        encoder.save(encoder_path)
        
    else:
        
        autoencoder = load_model(autoencoder_path)
        encoder = load_model(encoder_path)
        
        
        

    loss = autoencoder.evaluate(X, X, verbose=verbose)
    print('Evaluation loss: ', round(loss, 4), ' (MSE)')
    
    X_encoded = encoder.predict(X)
    print ('{} Representations Generated from {} Features'.format(X_encoded.shape[1], X.shape[1]))
    
    return X_encoded

def get_cm(tar, res):
    """
    Compare unsupervised learning results against known labels
    Homologous of comp_val in EDA_Functions

    :param tar: list, known labels
    :param res: list, results
    :return: dataframe, confusion matrix
    """
    cl = set(tar)
    cl = set(tar)
    result = [{x: 0 for x in cl} for i in range(max(res) + 1)]
    for i, c in enumerate(res):
        result[c][tar[i]] += 1
    return pd.DataFrame(result)

def labeled_metrics(X, y, model):
    """
    Metrics for supervised clustering

    :param y: Array of Arrays, Features Matrix
    :param model: object, clustering algorithm
    :return: Confusion matrix and fowlkes mallows score
    """

    np.set_printoptions(precision=2)
    labels = model.predict(X)

    return(get_cm(y, labels), round(fowlkes_mallows_score(y, labels), 3))

def unlabeled_metrics(X, model, name=''):
    """
    Metrics for unsupervised clustering

    :param X: Array of Arrays, Features Matrix
    :param model: object, clustering algorithm
    :return: silhouette fowlkes mallows score and plot
    """
    visualizer = silhouette_visualizer(X, model, name, colors='yellowbrick')
    visualizer.show()
    
    return round(visualizer.silhouette_score_, 3)

def get_k(X, model=None, _range=(1,15), metric='distortion', timings=False):
    
    if not model:
        model = KMeans()
    
    visualizer = KElbowVisualizer(
        model=model, 
        k=_range, 
        metric=metric, 
        timings=timings
    )
    
    visualizer.fit(X)
    visualizer.show()
    
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer 
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.xmeans import xmeans

def clustering(X, y, method, centers):
    
    method = str(method)
    amount_centers = int(centers)
    
    if method == 'Xmeans':
    
        # generate centers
        initializer = kmeans_plusplus_initializer(X, amount_centers).initialize()
        # Train X-Means Model
        xmeans_instance = xmeans(list(X), initializer, 20)
        xmeans_instance.process()
        # Extract results
        clusters = xmeans_instance.get_clusters()
        print("{} Clusters Generated".format(len(clusters)))
        # Print total sum of metric errors
        print("Total WCE: {}".format(round(xmeans_instance.get_total_wce(), 2)))
        
    # Unsupervised Metrics 
        
    score = unlabeled_metrics(X, xmeans_instance, name=method)
    print("Unsupervised Metrics: ")
    print("")
    print("Silhouette Score: {} (between [-1,1])".format(score))
    print('')
    print('')
    
    # Supervised Metrics
    
    cm, score = labeled_metrics(X, y, xmeans_instance)
    print('Supervised Metrics:')
    print('')
    print("Fowlkes Mallows Score: {} (between [0,1])".format(score))
    print('Confusion Matrix: \n', cm)