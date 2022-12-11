
# date: 2022-12-10

"""Creates exploratory data analysis figures from the preprocessed training data of the airbnb dataset (from https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

Saves the figures to the provided folder as .png files.

Usage: src/02-eda_figures.py --train_data=<train_data> --out_file=<out_file>

Example:

python src/02-eda_figures.py --train_data='data/processed/train.csv' --out_file='results'

Options:
--train_data=<train_data>   Path (including filename) of where the data is stored
--out_file=<out_file>       Path to directory where the figures will be saved
"""

# import required packages
from docopt import docopt
import os
import numpy as np
import pandas as pd
import altair as alt
from altair_saver import save
import vl_convert as vlc
alt.data_transformers.disable_max_rows()

opt = docopt(__doc__)


def save_chart(chart, filename, scale_factor=4.0):
    """
    Save an Altair chart using vl-convert

    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    """
    if filename.split(".")[-1] == "svg":
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split(".")[-1] == "png":
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")


# Function by Joel Ostblom


def main(data, out_file):
    """Function to create EDA figures for DSCI_573 group project
    Args:
        data (str): Path to cleaned training data
        out_file (str): Path where figures will be stored at
    """

    # create pandas dataframe of preprocessed train data
    train_data = pd.read_csv(data)
    
    #list of numeric features
    numeric_features = train_data.select_dtypes(include='number').columns.tolist()

    heat_map = alt.Chart(train_data).mark_rect(opacity = 0.5).encode(
        alt.X('longitude', bin=alt.Bin(maxbins=60)),
        alt.Y('latitude', bin=alt.Bin(maxbins=60)),
        alt.Color('reviews_per_month',scale=alt.Scale(scheme = 'oranges')))

    try:
        save_chart(heat_map, f"{out_file}/heat_map.png")
    except:
        os.makedirs(os.path.dirname(f"{out_file}/"))    
        save_chart(heat_map, f"{out_file}/heat_map.png")   
    
    #correlation of the numeric features 
    correlation = train_data[numeric_features].corr('pearson').style.background_gradient()
    save_chart(correlation, f"{out_file}/correlation.png")
    
    #distribution of numerical features
    charts = []
    even_chart = None
    odd_chart = None

    for count, column in enumerate(numeric_features):
        temp = alt.Chart(train_data).transform_density(
            column,
            as_=[column, 'density']
        ).mark_area().encode(
            alt.X(column, type='quantitative'),
            alt.Y('density:Q')
        ).properties(
            height=150,
            width=250
        )
        if count%2:
            if odd_chart != None:
                odd_chart = odd_chart & temp
            else:
                odd_chart = temp
        else:
            if even_chart != None:
                even_chart = even_chart & temp
            else:
                even_chart = temp
    num_chart = odd_chart | even_chart
   
    num_chart.properties(
        title="Distribution of Numeric Features"
    ).configure_title(anchor="middle")

    save_chart(num_chart, f"{out_file}/numeric_features.png")
    
    #log of numeric features
    log_numerical_features = ['price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',             
                             'availability_365']
    charts = []
    even_chart = None
    odd_chart = None

    for count, column in enumerate(log_numerical_features):
        temp = alt.Chart(train_df[log_numerical_features].apply(np.log)).transform_density(
            column,
            as_=[column, 'density']
        ).mark_area().encode(
            alt.X(column, type='quantitative'),
            alt.Y('density:Q')
        ).properties(
            height=150,
            width=250
        )
        if count%2:
            if odd_chart != None:
                odd_chart = odd_chart & temp
            else:
                odd_chart = temp
        else:
            if even_chart != None:
                even_chart = even_chart & temp
            else:
                even_chart = temp

    log_num_chart = odd_chart | even_chart 
    
    save_chart(log_num_chart, f"{out_file}/numeric_features(log_distribution).png")
    
    cat_chart = alt.Chart(data).mark_bar(opacity=0.7).encode(
                                                        alt.X(alt.repeat(), type='ordinal'),
                                                        alt.Y('count()', stack=False)
                                                        ).properties(height=150,width=600).repeat(
                                                        ['neighbourhood_group', 'room_type'],columns=1)
    
    save_chart(cat_chart, f"{out_file}/categorical_features.png")
    
    cat_charts = []

    for column in data['neighbourhood_group'].unique():
        temp = alt.Chart(data[data['neighbourhood_group'] == column]).mark_bar(opacity=0.7).encode(
            alt.X('count()',stack = False),
            alt.Y('neighbourhood', type = 'ordinal', sort='x')
        )
        cat_charts.append(temp)

    concat_charts = alt.vconcat(*cat_charts)
    save_chart(concat_charts, f"{out_file}/categorical_features(neighbourhood_group).png")
    

if __name__ == "__main__":
    main(opt["--train_data"], opt["--out_file"])
