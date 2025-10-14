import plotly
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


filename = "Evaluation_2025-03-25.csv"

df = pd.read_csv(filename,skiprows=5)

df["hovertext"] = "vars = "+df["vars"].astype(str)+ ", degree = " +df["degree"].astype(str)+ ", seed = " +df["seed"].astype(str)

df["Min"]=-1*df["Min"]

Types = list(df["Type"].drop_duplicates())

colorlist=["red","blue","green"]

pd_List =[]

for type in Types:
    df_part = df.mask(df["Type"]!=type).dropna()
    df_part = df_part.reset_index()
    pd_List.append(df_part)

fig = make_subplots(rows=2, cols=1,subplot_titles=("Run time","-Minimum"), shared_xaxes=True)

for i in range(len(pd_List)):
    fig.add_trace(
            go.Scatter(x=pd_List[i].index,y=pd_List[i]["Time"],legendgroup=Types[i],hovertext=pd_List[i]["hovertext"], mode='markers', 
                         marker=dict(symbol='x', color=colorlist[i], size=10), 
                         name=Types[i]),row=1,col=1)
    fig.add_trace(
            go.Scatter(x=pd_List[i].index,y=pd_List[i]["Min"],legendgroup=Types[i],hovertext=pd_List[i]["hovertext"], mode='markers', 
                         marker=dict(symbol='x', color=colorlist[i], size=10), 
                         name=Types[i]),row=2,col=1)
fig.update_layout(
    title_text=filename
)
fig.show()
fig.write_html(filename.split(".csv")[0]+".html")

