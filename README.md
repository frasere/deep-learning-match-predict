# Deep neural nets for football match prediction

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

This notebook predicts the outcome of football matches using historical match data. The test season is the 2019/2020 English Premier League (the prior 10 seasons across the top 5 European leagues are used for training).


## Program Description and Structure
I'm using _demo_nb.ipynb_ as the front end of this project. This explains in great detail the approach and displays the latest round of results that I had on the test set. The underlying models and data processing functions are in the _common_ folder. 


## How users get started
To install the dependencies,
```
pip install -r requirements.txt`
```
I have _not_ packaged everything up yet to enable installation via pip. Please clone the repo and explore the demo notebook it explains thoroughly what is going on. 

To get the match result data go to [this site](https://www.football-data.co.uk/data.php). Here you will find multiple seasons of football data covering pretty much all the leagues in the world. The data mainly consists of match results, total shots and betting odds. __Download the data and store in a folder within the project__. I have not included the raw data on the repo as its bad etiquette to do so! It doesn't matter how you store the data (i.e folder structure), just use the _load_all_matches_ function in the _data_methods.py_ module and pass your data folder location as the argument (it will search all the subfolder by default).

Then go ahead and run the notebook and make predictions!


## Maintenance and Support
Contact Fraser Ewing


## Copyright
See License file
