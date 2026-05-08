'''Sets up plot options'''

import matplotlib.pyplot as plt # type: ignore
from typing import Optional, Tuple

AX = plt.gca()
FIG = plt.gcf()

class PlotOptions(object):

    def __init__(self,
            ax=AX, 
            fig = FIG,   # type: ignore
            title: Optional[str] = None,
            xlabel: str = "time",
            ylabel: str = "concentration",
            legend: bool = True,
            xlim: Optional[Tuple[float, float]] = None,
            ylim: Optional[Tuple[float, float]] = None,
            ):
        if (AX == ax) and (FIG == fig):
            fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim

    def to_dict(self):
        return self.__dict__
    
    def apply(self):
        if self.title is not None:
            self.ax.set_title(self.title) # type: ignore
        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel)  # type: ignore
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel) # type: ignore
        if isinstance(self.legend, bool) and self.legend:
            self.ax.legend() # type: ignore
        if isinstance(self.legend, list):
            self.ax.legend(self.legend) # type: ignore
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim) # type: ignore
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)  # type: ignore