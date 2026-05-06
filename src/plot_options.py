'''Sets up plot options'''

import matplotlib.pyplot as plt # type: ignore
from typing import Optional, Tuple

class PlotOptions(object):

    def __init__(self,
            ax: Optional[plt.Axes] = None,   # type: ignore
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            legend: bool = True,
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            ):
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
    
    def apply(self):
        if self.title is not None:
            self.ax.set_title(self.title)
        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel)
        if self.legend:
            self.ax.legend()
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)