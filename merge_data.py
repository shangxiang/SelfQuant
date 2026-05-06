import pandas as pd
import glob
import numpy as np
from typing import List, Optional

raw_data_path = "data/raw/"
income_quarterly_columns = [
    'basic_eps', 'diluted_eps', 'total_revenue', 'revenue', 'int_income',
    'prem_earned', 'comm_income', 'n_commis_income', 'n_oth_income',
    'n_oth_b_income', 'prem_income', 'out_prem', 'une_prem_reser',
    'reins_income', 'n_sec_tb_income', 'n_sec_uw_income', 'n_asset_mg_income',
    'oth_b_income', 'fv_value_chg_gain', 'invest_income', 'ass_invest_income',
    'forex_gain', 'total_cogs', 'oper_cost', 'int_exp', 'comm_exp',
    'biz_tax_surchg', 'sell_exp', 'admin_exp', 'fin_exp', 'assets_impair_loss',
    'prem_refund', 'compens_payout', 'reser_insur_liab', 'div_payt',
    'reins_exp', 'oper_exp', 'compens_payout_refu', 'insur_reser_refu',
    'reins_cost_refund', 'other_bus_cost', 'operate_profit', 'non_oper_income',
    'non_oper_exp', 'nca_disploss', 'total_profit', 'income_tax', 'n_income',
    'n_income_attr_p', 'minority_gain', 'oth_compr_income', 't_compr_income',
    'compr_inc_attr_p', 'compr_inc_attr_m_s', 'ebit', 'ebitda', 'insurance_exp',
    'undist_profit', 'distable_profit', 'rd_exp', 'fin_exp_int_exp',
    'fin_exp_int_inc', 'transfer_surplus_rese', 'transfer_housing_imprest',
    'transfer_oth', 'adj_lossgain', 'withdra_legal_surplus',
    'withdra_legal_pubfund', 'withdra_biz_devfund', 'withdra_rese_fund',
    'withdra_oth_ersu', 'workers_welfare', 'distr_profit_shrhder',
    'prfshare_payable_dvd', 'comshare_payable_dvd', 'capit_comstock_div',
    'continued_net_profit'
]
cashflow_quarterly_columns = [
    'net_profit',
    'finan_exp',
    'c_fr_sale_sg',
    'recp_tax_rends',
    'n_depos_incr_fi',
    'n_incr_loans_cb',
    'n_inc_borr_oth_fi',
    'prem_fr_orig_contr',
    'n_incr_insured_dep',
    'n_reinsur_prem',
    'n_incr_disp_tfa',
    'ifc_cash_incr',
    'n_incr_disp_faas',
    'n_incr_loans_oth_bank',
    'n_cap_incr_repur',
    'c_fr_oth_operate_a',
    'c_inf_fr_operate_a',
    'c_paid_goods_s',
    'c_paid_to_for_empl',
    'c_paid_for_taxes',
    'n_incr_clt_loan_adv',
    'n_incr_dep_cbob',
    'c_pay_claims_orig_inco',
    'pay_handling_chrg',
    'pay_comm_insur_plcy',
    'oth_cash_pay_oper_act',
    'st_cash_out_act',
    'n_cashflow_act',
    'oth_recp_ral_inv_act',
    'c_disp_withdrwl_invest',
    'c_recp_return_invest',
    'n_recp_disp_fiolta',
    'n_recp_disp_sobu',
    'stot_inflows_inv_act',
    'c_pay_acq_const_fiolta',
    'c_paid_invest',
    'n_disp_subs_oth_biz',
    'oth_pay_ral_inv_act',
    'n_incr_pledge_loan',
    'stot_out_inv_act',
    'n_cashflow_inv_act',
    'c_recp_borrow',
    'proc_issue_bonds',
    'oth_cash_recp_ral_fnc_act',
    'stot_cash_in_fnc_act',
    'free_cashflow',
    'c_prepay_amt_borr',
    'c_pay_dist_dpcp_int_exp',
    'incl_dvd_profit_paid_sc_ms',
    'oth_cashpay_ral_fnc_act',
    'stot_cashout_fnc_act',
    'n_cash_flows_fnc_act',
    'eff_fx_flu_cash',
    'n_incr_cash_cash_equ',
    'c_recp_cap_contrib',
    'incl_cash_rec_saims',
    'uncon_invest_loss',
    'prov_depr_assets',
    'depr_fa_coga_dpba',
    'amort_intang_assets',
    'lt_amort_deferred_exp',
    'decr_deferred_exp',
    'incr_acc_exp',
    'loss_disp_fiolta',
    'loss_scr_fa',
    'loss_fv_chg',
    'invest_loss',
    'decr_def_inc_tax_assets',
    'incr_def_inc_tax_liab',
    'decr_inventories',
    'decr_oper_payable',
    'incr_oper_payable',
    'others',
    'im_net_cashflow_oper_act',
    'conv_debt_into_cap',
    'conv_copbonds_due_within_1y',
    'fa_fnc_leases',
    'im_n_incr_cash_equ',
    'net_dism_capital_add',
    'net_cash_rece_sec',
    'credit_impa_loss',
    'use_right_asset_dep',
    'oth_loss_asset'
]

series_list_name = [
    "balancesheet",
    "cashflow",
    "income",
    "stock_data",
]

section_list_name = [
    "daily_basic_data",
    "margin_detail",
    "money_flow",
    "top_list"
]


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理重复行：相同 (ts_code, f_ann_date, end_type) 只保留最佳行。
    最佳行定义：先按 ann_date 最新，再按非空列数最多。
    """
    # 计算每行非空值个数（仅数值列或全部列）
    df = df.copy()
    df['_non_null_count'] = df.select_dtypes(include=[np.number]).count(axis=1)
    #print(df[['_non_null_count', 'ts_code', 'f_ann_date', 'end_type']])

    # 按股票分组，然后对每个 (f_ann_date, end_type) 组内排序并取第一条
    def keep_best_group(g):
        g = g.sort_values(
            by=['ann_date', '_non_null_count'],
            ascending=[False, False],
            na_position='last'
        )
        return g.head(1)

    df_clean = (
        df.groupby(['ts_code', 'f_ann_date', 'end_type'], group_keys=False)
        .apply(keep_best_group)
        .drop(columns=['_non_null_count'])
        .reset_index(drop=True)
    )
    return df_clean


def convert_to_quarterly(df: pd.DataFrame, quarterly_cols) -> pd.DataFrame:
    """
    将累计财报值转换为单季度值。
    参数 quarterly_cols: 需要做季度化处理的列名列表（累计值）。
    对于每个股票，按 end_date 升序，对每列计算：单季 = 当前累计 - 上一报告期累计（同一年份内）。
    第一季度（03-31）保持不变。
    """
    #df = df.copy()
    # 确保 end_date 为日期类型，并提取年份和月份
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y%m%d')
    df['year'] = df['end_date'].dt.year
    df['month'] = df['end_date'].dt.month
    df['end_date'] = df['end_date'].dt.strftime('%Y%m%d')

    # 定义季度顺序映射 (按月份)
    df['quarter_order'] = df['month'].map({3: 1, 6: 2, 9: 3, 12: 4})
    # 保留必要列 sort 用
    df = df.sort_values(['ts_code', 'year', 'quarter_order'])

    # 对每个股票组，计算单季度值
    def compute_single_quarter(group):
        # group 中不含 ts_code（如果 include_groups=False）
        group = group.sort_values(['year', 'quarter_order'])
        for col in quarterly_cols:
            group[col] = pd.to_numeric(group[col], errors='coerce')
            prev_val = group[col].shift(1)
            # 使用 .loc 避免碎片化警告（但本质还是要加列）
            group.loc[:, col] = np.where(
                (group['quarter_order'] > 1) & (group['year'] == group['year'].shift(1)),
                group[col] - prev_val,
                group[col]
            )
        return group

    df = df.groupby('ts_code', group_keys=False).apply(compute_single_quarter)

    # 重命名：将 _quarterly 列替换原列（或保留原列，增加新列）
    rename_dict = {f"{col}_quarterly": col for col in quarterly_cols}
    df = df.rename(columns=rename_dict)
    # 删除中间辅助列
    df.drop(columns=['year', 'month', 'quarter_order'], inplace=True)
    return df

def financial_data_preprocess():
    """
    预处理下财务数据这种不是每天都有的数据类型
    :return:
    """

    # 先处理资产负债表，这个表里面都是时点性质的数据，不需要计算单季度
    all_balancecsheet = []
    for file in glob.glob(raw_data_path + "balancesheet/" + "*.csv"):
        df = pd.read_csv(file).drop(columns="Unnamed: 0")
        df = clean_duplicates(df) # 先处理后处理都一样
        all_balancecsheet.append(df)
    balancesheet_df = pd.concat(all_balancecsheet)
    balancesheet_df['end_date'] = pd.to_datetime(balancesheet_df['end_date'], format='%Y%m%d')
    balancesheet_df['end_date'] = balancesheet_df['end_date'].dt.strftime('%Y%m%d')

    balancesheet_df.to_csv("data/" + "balancesheet.csv")
    final_df = balancesheet_df # 借用下第一个表作为最终merge

    # 处理利润表，这个表里面需要折算到单季度
    all_income = []
    for file in glob.glob(raw_data_path + "income/" + "*.csv"):
        df = pd.read_csv(file).drop(columns="Unnamed: 0")
        df = clean_duplicates(df)  # 先处理后处理都一样
        df = convert_to_quarterly(df, quarterly_cols=income_quarterly_columns)
        all_income.append(df)

    income_df = pd.concat(all_income)
    income_df.to_csv("data/" + "income.csv")
    final_df = pd.merge(final_df, income_df, how='left', on=['ts_code', 'f_ann_date', 'end_date'])

    # 处理现金流表，这个表里面需要折算到单季度
    all_cashflow = []
    for file in glob.glob(raw_data_path + "cashflow/" + "*.csv"):
        df = pd.read_csv(file).drop(columns="Unnamed: 0")
        df = clean_duplicates(df)  # 先处理后处理都一样
        df = convert_to_quarterly(df, quarterly_cols=cashflow_quarterly_columns)
        all_cashflow.append(df)

    cashflow_df = pd.concat(all_cashflow)
    cashflow_df.to_csv("data/" + "cashflow.csv")
    final_df = pd.merge(final_df, cashflow_df, how='left', on=['ts_code', 'f_ann_date', 'end_date'])

    columes = final_df.columns.tolist()
    print(columes)
    final_df.to_csv("data/" + "financial.csv")

        #print(df[['ts_code', 'f_ann_date', 'end_type', 'end_date']])
        #df_final = convert_to_quarterly(df, quarterly_cols=income_quarterly_columns)
        #print(df_final[['ts_code', 'f_ann_date', 'end_type', 'end_date', 'basic_eps']])

def merge_basic_daily_data():
    """
    用于把每天都有数据的表整合起来，索引为ts_code和trade_date
    :return:
    """
    basic_daily_data_list = [
        "stock_data",
        "daily_basic_data",
        "margin_detail",
        "moneyflow",
        "top_list"
    ]
    final_result = None
    for section_name in basic_daily_data_list:
        all_daily = []
        for file in glob.glob(raw_data_path + section_name + "/*.csv"):
            df = pd.read_csv(file).drop(columns="Unnamed: 0")
            all_daily.append(df)

        daily_df = pd.concat(all_daily)
        daily_df = daily_df.set_index(["ts_code", "trade_date"]).sort_index()
        daily_df.to_csv("data/series/" + section_name + ".csv")
        print("尝试把", section_name, "merge到基础表中")
        if final_result is None:
            final_result = daily_df
        else:
            final_result = pd.merge(final_result, daily_df, how="left", on=["ts_code","trade_date"])
        del daily_df
    final_result.to_csv("data/section/final_result.csv")


def split_to_series_section(input_file_name):
    """
    用于把全部merge到一起的文件拆分成时序的和截面的
    :param file_name:
    :return:
    """
    series_path = "data/series/"
    df = pd.read_csv(input_file_name)
    print(df.columns)
    for ts_code, group in df.groupby("ts_code"):
        group = group.sort_values("trade_date") if "trade_date" in group.columns else group
        output_file_name = series_path + str(ts_code) + ".csv"
        group.to_csv(output_file_name, index=False)
        print("已保存", str(ts_code))


if __name__ == '__main__':
    split_to_series_section("data/final_result.csv")
