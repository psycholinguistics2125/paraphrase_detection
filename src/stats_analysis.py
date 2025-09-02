import pingouin as pg
import pandas as pd
import itertools
from scipy.stats import ranksums
from pingouin import power_ttest2n
from pingouin import power_anova
import seaborn as sns

def compute_wilcoxon(data: pd.DataFrame, x: str, y: str, alt="two-sided") -> dict:
    """compute wilcowon test beween the col x and y

    Args:
        data (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        alt (str, optional): _description_. Defaults to 'two-sided'.

    Returns:
        dict: _description_
    """
    a = data[data[x] == 1][y]
    b = data[data[x] == 0][y]
    t = ranksums(a, b, alternative=alt)
    stats = {}
    stats["x"] = x
    stats["y"] = y
    stats["stats"] = t[0]
    stats["pval"] = t[1]
    stats["cohen"] = pg.compute_effsize(a, b, eftype="cohen")
    stats["power"] = power_ttest2n(nx=len(a), ny=len(b), d=stats["cohen"], alpha=0.01)
    return stats


def compute_wilc_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    i = 0
    df_result = pd.DataFrame(columns=["x", "y", "stats", "pval", "cohen", "power"])
    for elt in list(itertools.product(cible, col_list)):
        try:
            x = elt[0]
            y = elt[1]
            result = compute_wilcoxon(data, x, y)
            if result["pval"] <= seuil:
                df_result.loc[i] = result
                i = i + 1
        except:
            continue
    return df_result


def compute_anova_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    i = 0
    df_result = pd.DataFrame(columns=["x", "y", "p-unc", "np2", "power"])
    for elt in list(itertools.product(cible, col_list)):
        result = {}
        x = elt[0]
        y = elt[1]
        
        result["x"] = x
        result["y"] = y
        try:
            stats = pg.anova(data=data, dv=y, between=x, detailed=True).to_dict(
            orient="records")[0]
            result["p-unc"] = stats["p-unc"]
            result["np2"] = stats["np2"]
            result["power"] = power_anova(
                eta_squared=result["np2"], k=len(set(data[x].tolist())), n=len(data), alpha=0.05
            )
        except:
            result["p-unc"] = 1
            result["np2"] = 0
        if result["p-unc"] <= seuil:
            df_result.loc[i] = result
            i = i + 1

    return df_result


def compute_tttest(data: pd.DataFrame, x: str, y: str) -> dict:
    a = data[data[y] == 0][x]
    b = data[data[y] == 1][x]

    return (pg.ttest(a, b, correction=False).to_dict(orient="records")[0])


def compute_ttest_table(data: pd.DataFrame, cible: list, col_list: list, seuil=0.01) -> pd.DataFrame:
    i = 0
    df_result = pd.DataFrame(columns=["x", "y", 'T', 'dof', 'alternative', 'p-val', 'CI95%', 'cohen-d', 'BF10',
                                      'power'])
    for elt in list(itertools.product(cible, col_list)):
        try:

            x = elt[0]
            y = elt[1]

            result = compute_tttest(data, y, x)
            result['x'] = x
            result['y'] = y

            if result['p-val'] <= seuil:
                df_result.loc[i] = result
                i = i+1
        except Exception as e:

            #print(e)
            continue
    return df_result


def compute_pearson_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01, method="pearson"
) -> pd.DataFrame:
    """compute mann whitney u test for two sets of columns

    Args:
        data (pd.DataFrame): _description_
        cible (list): _description_
        col_list (list): _description_
        seuil (float, optional): _description_. Defaults to 0.01.

    Returns:
        pd.DataFrame: _description_
    """
    i = 0
    df_result = pd.DataFrame(
        columns=[
            "x",
            "y",
            "n",
            "r",
            "CI95%",
            "p-val",
            "BF10",
            "power",
        ]
    )
    for elt in list(itertools.product(cible, col_list)):
        try:
            x = elt[0]
            y = elt[1]
            #print(x,y)
            result = pg.corr(data[x].apply(float).values,data[y].apply(float).values, method=method)
            #print(result)
            result["x"] = x
            result["y"] = y
            
            if result["p-val"].values[0]<= seuil:
                #print(result)
                df_result.loc[i] = pd.Series(result.to_dict(orient="index")[method])
                i = i + 1
        except Exception as e:
            result = {
                "x": elt[0],
                "y": elt[1],
                "p-val": 0.2,
                "r": 0,
                "CI95%": [0, 0],
                "BF10": "failed",
                "power": 0,
                "n": 0
            }
            df_result.loc[i] = pd.Series(result)
            i = i + 1


            print(e,x,y)
            continue
    return df_result