# Setup
import os
import chart_studio
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import plotly.graph_objects as go
import datapane as dp

GREY = '#788995'
BLUE = '#0059ff'
GOLD = '#fdbd28'
GREEN = '#28D9AA'
RED = '#EE5149'

def sim_data():
    data_1 = make_blobs(n_samples=[200], n_features=2, centers=[(0,0)],
                      cluster_std=0.4, random_state=12)
    data_1 = pd.DataFrame(data_1[0])
    data_1["outlier"] = 1
    outliers = {0:[-2, 1.7, 1.4, -1.8, 2, 0], 1:[-2, -1, 1.2, -1.5, 1.4, 1.5],
           "outlier":-1}
    data_1 = data_1.append(pd.DataFrame(outliers))
    data_1.columns = ["x","y","outlier"]

    data_2 = make_blobs(n_samples=[120, 50, 30], n_features=2, centers=[(-1,-1), (0, 1), (1.5,1.3)],
                      cluster_std=[0.3, 0.2, 0.2], random_state=12)
    data_2 = pd.DataFrame(data_2[0])
    data_2["outlier"] = 1
    outliers = {0:[-2, 2, 2.1, -1.8, 2.5, 0], 1:[-2, -1, 2, 2.2, 1.7, 2.3],
               "outlier":-1}
    data_2 = data_2.append(pd.DataFrame(outliers))
    data_2.columns = ["x","y","outlier"]
    
    return data_1, data_2

def lof_for_different_n(data_1, data_2):
    X = data_1.loc[:,("x","y")].values
    outclass = []

    for i in range(5, 205, 5):
        clf = LocalOutlierFactor(n_neighbors=i, contamination="auto")
        y_pred = clf.fit_predict(X)
        score = clf.negative_outlier_factor_
        radius = (score.max() - score) / (score.max() - score.min())
        outclass += [np.array([y_pred, [i]*len(y_pred), score, radius]).T]

    id = np.arange(1,len(X)+1,1)
    df_1 = pd.DataFrame(np.concatenate(outclass), columns = ("outlier_pred", "neighbors",
                                                           "lof", "radius"))
    df_1["outlier_pred_n"] = np.where(df_1["outlier_pred"]==1, "Not an Outlier", "Outlier")
    coords = pd.DataFrame(np.tile(X, (40,1)), columns=("x","y"))
    df_interact_1 = pd.concat([coords,df_1], axis=1)
    df_interact_1["uid"] = np.tile(id, 40)
    df_interact_1['centers'] = "One Center"

    # ------------------------------------

    X = data_2.loc[:,("x","y")].values
    outclass = []

    for i in range(5, 205, 5):
        clf = LocalOutlierFactor(n_neighbors=i, contamination="auto")
        y_pred = clf.fit_predict(X)
        score = clf.negative_outlier_factor_
        radius = (score.max() - score) / (score.max() - score.min())
        outclass += [np.array([y_pred, [i]*len(y_pred), score, radius]).T]

    id = np.arange(len(X)+1,2*len(X)+1,1)
    df_2 = pd.DataFrame(np.concatenate(outclass), columns = ("outlier_pred", "neighbors",
                                                           "lof", "radius"))
    df_2["outlier_pred_n"] = np.where(df_1["outlier_pred"]==1, "Not an Outlier", "Outlier")
    coords = pd.DataFrame(np.tile(X, (40,1)), columns=("x","y"))
    df_interact_2 = pd.concat([coords,df_2], axis=1)
    df_interact_2["uid"] = np.tile(id, 40)
    df_interact_2['centers'] = "Two Centers"

    df_viz = pd.concat([df_interact_1, df_interact_2])
    
    return df_viz

def make_interactive_fig(df):
    fig = px.scatter(df_viz, x="x", y="y",  
                     color="outlier_pred_n",
                     animation_frame="neighbors",
                     animation_group="uid",
                     title="<b>Local Outlier Factor with Simulated Data</b>",
                     size = "radius",
                     size_max=40,
                     symbol_sequence=["circle-open-dot"],
                     color_discrete_sequence=[BLUE, RED],
                     facet_col = "centers",
                     labels={
                         "x": "X",
                         "y": "Y",
                         "outlier_pred_n": "Outlier Status",
                         "neighbors": "Number of Neighbors",
                     "radius": "radius"},
                    hover_data = {"radius":False,
                                 "centers":False})
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("centers=", "")))
    fig["layout"].pop("updatemenus")
    fig.update_traces(marker={"line":{"width":2}})
    fig.update_xaxes(range=[-2.5, 2.5])
    fig.update_yaxes(range=[-2.5, 3])

    fig.update_layout(
        title={
            'font':{'size':18},
            'text': "Outlier Detection with LOF<br>Depends on Number of Neighbors",
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        margin=dict(l=50, r=50, t=50, b=10),
        legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
    )

    fig.add_annotation(x=0, y=-1.5,
                text="<b>Cluster 1</b>:<br>Center: (0,0)<br>N=200<br>SD: 0.4",
                showarrow=False)

    fig.add_annotation(x=-0.5, y=-2.1, xref="x2",
                text="<b>Cluster 1</b> :<br>Center: (-1,1.5))<br>N=120<br>SD: 0.3",
                showarrow=False)

    fig.add_annotation(x=-0.25, y=0.18, xref="x2",
                text="<b>Cluster 2</b>:<br>Center: (0,1))<br>N=50<br>SD: 0.2",
                showarrow=False)

    fig.add_annotation(x=1.5, y=0.55, xref="x2",
                text="<b>Cluster 3</b>:<br>Center: (1.5,1.3)<br>N=30<br>SD: 0.2",
                showarrow=False)

    fig.add_annotation(x=0, y=0.98, xref="paper", yref="paper",
                text="Circle radii are proportional to Local Outlier Factor score",
                showarrow=False, align="left", )

    fig['layout']['sliders'][0]['pad']=dict(l=-50, r=0, b=0, t=20)

    fig.update_layout(
        autosize=False,
        width=900, height=600,
        font_family="Iosevka Term",
        font_color="black",
        title_font_family="Roboto Slab",
        title_font_color="black",
        legend_title_font_color="black"
    )
    
    return fig

def upload_to_datapane(fig):
    report = dp.Report(dp.Plot(fig, name="neighbors", responsive=False) ) 
    report.upload(name='lof_neighbors', open=True)

    
if __name__ == '__main__':
    data_1, data_2 = sim_data()
    df_viz = lof_for_different_n(data_1=data_1, data_2=data_2)
    interactive = make_interactive_fig(df_viz)
    
    upload_to_datapane(interactive)