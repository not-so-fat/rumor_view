# rumor_view

Python library to visualize categorical column values relationship.

# Objective

Help explore data frame to find column values which has strong relationship.
This module estimate P(x1=v1 | x2=v2), and consider significance based on
"lift" := P(x1=v1 | x2=v2) / P(x1=v1).

There are two main functionalities;

- `show_columns`
    - This method provides a map for your exploration. This visualize graph which node represents each column, and edge between nodes when correspondent columns have lift in some value pair of those columns.
- `show_relation`
    - This method provides a detailed view for a pair of columns. With this you can see concrete values which has strong relationship

# Installation

```
pip install rumor_view
```

# Usage

Right now this module handles all the column as categorical. If you have numeric columns it is recommended to convert them as categorical values before inputing this module.

```
import rumor_view


view = rumor_view.RumorView(df)

view.show_columns() # this shows all the columns
view.show_columns(target_column="target") # this shows only 2 hops from target column

view.show_relations("target", "column") # this shows value relationships between input columns
```
