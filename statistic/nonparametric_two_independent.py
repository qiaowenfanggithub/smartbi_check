# -*- coding: utf-8 -*-
import pandas as pd
import scipy.stats as stats
import logging

log = logging.getLogger(__name__)


# wilcoxon 秩和检验
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ranksums.html
def wilcoxon_ranksums_test(x1, x2):
    log.info('use Wilcoxon秩和检验')
    wilcoxon_ranksums_statistic, wilcoxon_ranksums_p = stats.ranksums(x1, x2)
    log.info('双边检测')
    log.info(wilcoxon_ranksums_statistic, wilcoxon_ranksums_p)
    return [{"title": "Wilcoxon秩和检验"},
            {"row": "双边检测"},
            {"col": ["统计量", "P值"]},
            {"data": [wilcoxon_ranksums_statistic, wilcoxon_ranksums_p]}]


def mannwhitneyu_test(x1, x2):
    log.info('use Mann-Whitney U 检验')
    alter = ['two-sided', 'greater', 'less']
    res = []
    for i in alter:
        if i == 'two-sided':
            mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative=i)
            log.info('双边检测：')
            log.info(mannwhitneyu_statistic, mannwhitneyu_pvalue)
            res.append((mannwhitneyu_statistic, mannwhitneyu_pvalue))
        elif i == 'greater':
            mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative=i)
            log.info('单侧检测，备择假设为">"：')
            log.info(mannwhitneyu_statistic, mannwhitneyu_pvalue)
            res.append((mannwhitneyu_statistic, mannwhitneyu_pvalue))
        elif i == 'less':
            mannwhitneyu_statistic, mannwhitneyu_pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative=i)
            log.info('单侧检测，备择假设为"<"：')
            log.info(mannwhitneyu_statistic, mannwhitneyu_pvalue)
            res.append((mannwhitneyu_statistic, mannwhitneyu_pvalue))
    return [{"title": "Mann-Whitney U 检验"},
            {"row": ["双边检测", "单侧检测，备择假设为'>'", "单侧检测，备择假设为'<'"]},
            {"col": ["统计量", "P值"]},
            {"data": res}]


def nonparam_two_independent_describe_info(data, X, Y):
    data_groupby = data.groupby(X)
    new_data = pd.concat([data_groupby[Y[0]].count(), data_groupby[Y[0]].mean(),
                          data_groupby[Y[0]].std(),
                          data_groupby[Y[0]].std() / data_groupby[Y[0]].count()], axis=1)
    new_data.columns = ["count", "mean", "std", "std_err"]
    return {
        "row": new_data.index.values.tolist(),
        "col": new_data.columns.values.tolist(),
        "data": new_data.values.tolist(),
    }


if __name__ == '__main__':
    data = pd.read_csv('./data/nonparametric_two_independent.csv')

    x1 = data['x1']
    x2 = data['x2']

    # use Wilcoxon秩和检验
    print(wilcoxon_ranksums_test(x1, x2))

    # use Mann-Whitney U 检验
    print(mannwhitneyu_test(x1, x2))
