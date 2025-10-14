import plotly
import csv 
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import ast
import numpy as np
df = px.data.tips()
fig = make_subplots(rows=1, cols=4)

filename = "data_thesis_JO.txt"

df= pd.read_csv(filename)

#df['Mins'] = [x.strip('[]').split(',') for x in df['Mins']]
print(df)

# Convert to long-form
#long_df = df.explode('Mins').reset_index(drop=True)
# Ensure values are numeric
#long_df['Mins'] = pd.to_numeric(long_df['Mins'])
# Plot
for index, row in df.iterrows():
    y=df["Mins"][index]
    y=eval(y, {"np": np})
    #y =  eval(y,{"array": np.array})
    if np.mod(index,3)==0:
        marker_color = "purple"
        name = "   "
    elif np.mod(index,3)==1:
        marker_color = "blue"
        name = "  "
    elif np.mod(index,3)==2:
        marker_color= "red"
        name = " "
    fig.add_trace(
        go.Box(y=y,marker_color=marker_color,name=name),col=int((index+3)/3),row=1
    )
    fig.update_layout(
        showlegend = False
    )
if 0:
    for i in range(1):
        fig.update_yaxes(
            tickfont=dict(size=17)
        )
if 0:
    fig.update_layout(
        xaxis=dict(title=dict(font=dict(size=20)),tickfont=dict(size=16)),
        yaxis=dict(title=dict(font=dict(size=20)),showticklabels=False),
        legend=dict(font=dict(size=20))
    )
        
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks='',
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks='',
            title=None
        ),
        showlegend=False,       # Hides the legend
        plot_bgcolor='white',   # Removes gray background
        paper_bgcolor='white',  # Removes outer background
        margin=dict(l=0, r=0, t=0, b=0)  # Optional: remove all margins
    )

fig.show()