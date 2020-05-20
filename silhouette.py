
import numpy as np
import matplotlib.ticker as ticker

from sklearn.metrics import silhouette_score, silhouette_samples

from yellowbrick.utils import check_fitted
from yellowbrick.style import resolve_colors
from matplotlib import pyplot as plt


class silhouette_visualizer():
    
    """
    Draw visualiation for silhouette metric.
    Adapted from https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html#silhouette-visualizer
    This visualiation follows the documentation in the link above, with some changes to use pyclustering models instead of sklearn models.
    All inheritances removed
    """

    def __init__(self, X, model, name='', colors=None, **kwargs):

        self.colors = colors
        if "colormap" in kwargs:
            self.colors = kwargs["colormap"]
            
        # Get the properties of the dataset
        self.n_samples_ = X.shape[0]

        # Compute the scores of the cluster
        try:
            labels = model.labels_
            self.n_clusters_ = len(set(labels))
        except AttributeError:
            labels = model.predict(X)
            self.n_clusters_ = len(model.get_clusters())
                
        self.silhouette_score_ = silhouette_score(X, labels)
        self.silhouette_samples_ = silhouette_samples(X, labels)
        
        self.ax = plt.gca()
        self.name = name
        

        # Draw the silhouette figure
        self.draw(labels)
        
    def show(self):

        self.finalize()
        plt.show()


    def draw(self, labels):

        # Track the positions of the lines being drawn
        y_lower = 10  # The bottom of the silhouette

        # Get the colors from the various properties
        color_kwargs = {"n_colors": self.n_clusters_}

        if self.colors is None:
            color_kwargs["colormap"] = "Set1"
        elif isinstance(self.colors, str):
            color_kwargs["colormap"] = self.colors
        else:
            color_kwargs["colors"] = self.colors

        colors = resolve_colors(**color_kwargs)

        # For each cluster, plot the silhouette scores
        self.y_tick_pos_ = []
        for idx in range(self.n_clusters_):

            # Collect silhouette scores for samples in the current cluster .
            values = self.silhouette_samples_[labels == idx]
            values.sort()

            # Compute the size of the cluster and find upper limit
            size = values.shape[0]
            y_upper = y_lower + size

            color = colors[idx]
            self.ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                facecolor=color,
                edgecolor=color,
                alpha=0.5,
            )

            # Collect the tick position for each cluster
            self.y_tick_pos_.append(y_lower + 0.5 * size)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        # The vertical line for average silhouette score of all the values
        self.ax.axvline(
            x=self.silhouette_score_,
            color="red",
            linestyle="--",
            label="Average Silhouette Score",
        )

        return self.ax


    def finalize(self):


        # Set the title
        plt.title(
            ("Silhouette Plot of {} Clustering for {} Samples in {} Centers").format(
                self.name, self.n_samples_, self.n_clusters_
            )
        )

        l_xlim = max(-1, min(-0.1, round(min(self.silhouette_samples_) - 0.1, 1)))
        u_xlim = min(1, round(max(self.silhouette_samples_) + 0.1, 1))
        self.ax.set_xlim([l_xlim, u_xlim])

        # The (n_clusters_+1)*10 is for inserting blank space between
        # silhouette plots of individual clusters, to demarcate them clearly.
        self.ax.set_ylim([0, self.n_samples_ + (self.n_clusters_ + 1) * 10])

        # Set the x and y labels
        self.ax.set_xlabel("silhouette coefficient values")
        self.ax.set_ylabel("cluster label")

        # Set the ticks on the axis object.
        self.ax.set_yticks(self.y_tick_pos_)
        self.ax.set_yticklabels(str(idx) for idx in range(self.n_clusters_))
        # Set the ticks at multiples of 0.1
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        # Show legend (Average Silhouette Score axis)
        self.ax.legend(loc="best")