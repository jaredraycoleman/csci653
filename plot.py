import plotly.express as px
import pandas as pd
import pathlib 


thisdir = pathlib.Path(__file__).resolve().parent

def main():
    # columns = ["graph", "app", "num_nodes", "num_edges", "num_patterns", "num_pattern_instances"]
    df = pd.read_csv(thisdir / 'pegasus-patterns.csv')
    
    # num_nodes x time plot
    fig = px.line(
        df, x='num_nodes', y='time', line_dash='method',
        facet_col='app', facet_col_wrap=3, facet_row_spacing=0.1, facet_col_spacing=0.1,
        labels={'num_nodes': 'Number of Nodes', 'time': 'Time (s)'},
        title='Time to Find All Pattern Instances',
        template='plotly_white'
    )
    # set line color to black
    fig.for_each_trace(lambda trace: trace.update(line_color='black'))
    # make hd resolution
    fig.update_layout(width=1920, height=1080)

    # increase font size
    fig.update_layout(font=dict(size=18), title_x=0.5)

    # independent axes
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)

    # save image
    fig.write_image('pegasus-patterns.png')
    fig.write_html('pegasus-patterns.html')

if __name__ == "__main__":
    main()