import copy
import logging
import numpy
from scipy.sparse import csc_array
import pandas
import networkx as nx
from pyvis.network import Network


logger = logging.getLogger(__name__)
SIZE=500
NX_OPTIONS_DEFAULT = dict(
    height=f'{SIZE}px', width=f'{SIZE}px', bgcolor='#05131e', font_color='white', notebook=True, cdn_resources='in_line'
)


class RumorView(object):
    """
    Module to help you explore column value relationships
    """
    def __init__(self, df):
        """
        Args:
            df (pandas.DataFrame): Dataframe you want to analyze column value relationships. All the columns are considered as categorical. If you have numerical columns it is recommended to convert it categorical.
        """
        self.n_nodes = 0
        self.column_names = df.columns
        self.node_names = []
        self.index_dict = {}
        self._make_index(df)
        self.freq_matrix = self._calc_freq_matrix(df)
        self.co_matrix = self.freq_matrix.transpose().dot(self.freq_matrix).todense()
        self.cprob_df = pandas.DataFrame(
            1 / self.co_matrix.diagonal() * self.co_matrix, 
            index=self.node_names, columns=self.node_names
        )
        self.node_prob = self.freq_matrix.sum(axis=0) / len(df)
        self.lift_df = pandas.DataFrame(
            numpy.diag((1 / self.node_prob)).dot(self.cprob_df.values),
            index=self.node_names, columns=self.node_names
        )
        
    def show_columns(
        self, target_column="", n_hops=2, columns=None, min_lift=1.5, 
        min_size=0.01, physics=False, **nx_options
    ):
        """
        Visualize columns to help you understand which pair would have relationships
        Args:
            target_column (str): if specified, result contains only nodes within `n_hops` from specified column
            n_hops (int): if `target_column` is specified this specifies how far nodes from `target_column` should be displayed.
            columns (list[str]): if specified, result contains only specified columns
            min_lift (float): minimum lift value to show edge. default=1.5
            min_size (float): minimum probability of column value to be considered for lift. default=0.01
            physics (bool): whether to use physics model for visualization results. default=False
        """
        column_summary = self._create_column_summary(min_size)
        if columns is None:
            columns = self.column_names
        column_summary = column_summary[
            (column_summary["from"].isin(columns))&(column_summary["to"].isin(columns))
        ]

        nxg = nx.Graph()
        for c in columns:
            color = "#fe6708" if c == target_column else "#a7c0f7"
            nxg.add_node(c, title=f"{c}\n cardinality={len(self.index_dict[c])}", color=color)
        for row_ind, row in column_summary.iterrows():
            if row["max(lift)"] >= min_lift:
                nxg.add_edge(
                    row["from"], row["to"], title=f'lift={row["max(lift)"]:.1%}', value=row["max(lift)"]
                )
        print(f"showing edges with lift >= {min_lift}")
        if target_column:
            nxg = nx.generators.ego_graph(nxg, target_column, n_hops)
            print(f"showing nodes within {n_hops} edges from {target_column}")
        nx_args = _overwrite_dict(NX_OPTIONS_DEFAULT, nx_options)
        g = Network(**nx_args)
        g.from_nx(nxg)
        g.toggle_physics(physics)
        return(g.show("column_plot.html"))
    
    def show_relations(self, c1, c2, min_lift=1.5, show_df=True, **nx_options):
        """
        Visualize value relationship between columns
        Args:
            c1 (str): column for condition
            c2 (str): column for result
            min_lift (float): minimum lift value to show edge. default=1.5
            show_df (bool): whether to output DataFrame of probability calculated
        """
        nx_args = _overwrite_dict(NX_OPTIONS_DEFAULT, nx_options)
        g = Network(directed=True, **nx_args)
        c1_ind = sorted(self.index_dict[c1].values())
        c1_nodes = [self.node_names[c] for c in c1_ind]
        c2_ind = sorted(self.index_dict[c2].values())
        c2_nodes = [self.node_names[c] for c in c2_ind]
        for i, (c1_i, c1_node) in enumerate(zip(c1_ind, c1_nodes)):
            g.add_node(
                c1_node, c1_node, title=f"{c1_node}\n size={self.node_prob[c1_i]:.1%}",
                x=-SIZE/2, y=i/len(c1_nodes)*SIZE, value=self.node_prob[c1_i]
            )
        for i, (c2_i, c2_node) in enumerate(zip(c2_ind, c2_nodes)):
            g.add_node(
                c2_node, c2_node, title=f"{c2_node}\n size={self.node_prob[c2_i]:.1%}",
                x=SIZE/2, y=i/len(c2_nodes)*SIZE, value=self.node_prob[c2_i]
            )
        for c1_node in c1_nodes:
            for c2_node in c2_nodes:
                if self.lift_df.loc[c2_node, c1_node] > min_lift:
                    title = f"{c1_node}->{c2_node}\n"
                    title += f"prob={self.cprob_df.loc[c2_node, c1_node]:.1%}\n"
                    title += f"lift={self.lift_df.loc[c2_node, c1_node]:.1%}"
                    g.add_edge(
                        c1_node, c2_node, title=title, value=self.lift_df.loc[c2_node, c1_node]
                    )
        g.toggle_physics(False)
        if show_df:
            display(self.cprob_df.loc[c2_nodes, c1_nodes])
        print(f"showing edges with lift >= {min_lift}")
        return(g.show("relations.html"))
        
        
    def _make_index(self, df):
        for c in df.columns:
            self.index_dict[c] = {}
            values = sorted(df[c].unique()) # change order based on numeric / categorical
            for v in values:
                self.index_dict[c][v] = self.n_nodes
                self.node_names.append(f"{c}-{v}")
                self.n_nodes += 1
    
    def _calc_freq_matrix(self, df):
        freq_matrix = csc_array((len(df), self.n_nodes))
        for i, ind in enumerate(df.index):
            for c in df.columns:
                j = self.index_dict[c][df.loc[ind, c]]
                freq_matrix[i, j] = 1
        return freq_matrix
    
    def _extract_matrix_for_2columns(self, df, c1, c2):
        c1_ind = list(self.index_dict[c1].values())
        c2_ind = list(self.index_dict[c2].values())
        return df.iloc[c1_ind, c2_ind]
    
    def _create_column_summary(self, min_size):
        from_list = []
        to_list = []
        lift_list = []
        for c1 in self.index_dict.keys():
            c1_ind = sorted(self.index_dict[c1].values())
            c1_nodes = [self.node_names[c] for c in c1_ind if self.node_prob[c] > min_size]
            for c2 in self.index_dict.keys():
                if c1 == c2:
                    continue
                c2_ind = sorted(self.index_dict[c2].values())
                c2_nodes = [self.node_names[c] for c in c2_ind if self.node_prob[c] > min_size]
                lift_df_2c = self.lift_df.loc[c2_nodes, c1_nodes]
                max_lift = lift_df_2c.max().max()
                from_list.append(c1)
                to_list.append(c2)
                lift_list.append(max_lift)
        return pandas.DataFrame({
            "from": from_list,
            "to": to_list,
            "max(lift)": lift_list
        })
    

def _overwrite_dict(d1, d2):
    ret_dict = copy.deepcopy(d1)
    for k, v in d2.items():
        ret_dict[k] = v
    return ret_dict
