import numpy as np
import pandas as pd




def main():
   
    df_dataset_missing01 = pd.read_csv("dataset_missing01.csv")
    df_dataset_missing20 = pd.read_csv("dataset_missing20.csv")
    df_dataset_complete = pd.read_csv("dataset_complete.csv")

    # 1
    mean_imputed_01 = mean(df_dataset_missing01.copy(), False, "V00825375_missing01_imputed_mean.csv")
    mean_imputed_20 = mean(df_dataset_missing20.copy(), False, "V00825375_missing20_imputed_mean.csv")

    # 2
    mean_imputed_cond_01 = mean(df_dataset_missing01.copy(), True, "V00825375_missing01_imputed_mean_conditional.csv")
    mean_imputed_cond_20 = mean(df_dataset_missing20.copy(), True, "VOO825375_missing20_imputed_mean_conditional.csv")

    # 3
    hotval01 = hotdeck(df_dataset_missing01.copy(), False, "V00825375_missing01_imputed_hd.csv")
    hotval20 = hotdeck(df_dataset_missing20.copy(), False, "VOO825375_missing20_imputed_hd.csv")

    # 4
    hotval_conditional01 = hotdeck(df_dataset_missing01.copy(), True, "V00825375_missing01_imputed_hd_conditional.csv")
    hotval_conditional20 = hotdeck(df_dataset_missing20.copy(), True, "V00825375_misisng20_imputed_hd_conditional.csv")



    ###Printing out MAE scores###
    maeVals01 = mae(df_dataset_missing01, mean_imputed_01, df_dataset_complete)
    print("MAE_01_mean=", maeVals01)
    maeVals01Cond = mae(df_dataset_missing01, mean_imputed_cond_01, df_dataset_complete)
    print("MAE_01_mean_conditional=", maeVals01Cond)
    hotVals01Print=mae(df_dataset_missing01, hotval01, df_dataset_complete)
    print("MAE_01_hd=", hotVals01Print)
    hotValsCond01P = mae(df_dataset_missing01, hotval_conditional01, df_dataset_complete)
    print("MAE_01_hd_conditional=",hotValsCond01P)

    maeVals20 = mae(df_dataset_missing20, mean_imputed_20, df_dataset_complete)
    print("MAE_20_mean=", maeVals20)
    maeVals20Cond = mae(df_dataset_missing20, mean_imputed_cond_20, df_dataset_complete)
    print("MAE_20_mean_conditional=", maeVals20Cond)
    hotVals20Print = mae(df_dataset_missing01,hotval20, df_dataset_complete)
    print("MAE_20_hd=", hotVals20Print)
    hotValsCond20P = mae(df_dataset_missing01, hotval_conditional20, df_dataset_complete)
    print("MAE_20_hd_conditional=", hotValsCond20P)


def hotdeck(df, conditional, filename):
    col_range = range(df.shape[1])
    row_range = range(df.shape[0])
    values = df.to_numpy()
    imputes = []
    for row in row_range:
        for col in col_range:
            if col >= 10:
                continue
            if values[row][col] is '?':
                missing_col = col
                if missing_col is not -1:
                    lowest_diff = 999999
                    target = -1
                    for row_2 in row_range:
                        if row_2 is row:
                            continue
                        if conditional and values[row][10] is not values[row_2][10]:
                            continue
                        diff = 0
                        diff_count = 0
                        for col2 in col_range:
                            if col2 >= 10:
                                continue
                            if values[row][col2] is '?' or values[row_2][col2] is '?' or values[row_2][missing_col] is '?':
                                continue
                            diff += abs(float(values[row][col2]) - float(values[row_2][col2]))
                            diff_count += 1
                        if diff_count is 0:
                            continue
                        if (diff / diff_count) < lowest_diff:
                            lowest_diff = diff / diff_count
                            target = values[row_2][missing_col]
                    imputes.append([row, missing_col, target])
    for impute in imputes:
        df.iat[impute[0], impute[1]] = impute[2]
    df.to_csv(filename)
    return df

def mean(df, conditional, filename):
    col_range = range(df.shape[1])
    row_range = range(df.shape[0])
    values = df.to_numpy()
    for col in col_range:
        if col >= 10:
            continue
        total_no = 0
        count_no = 0
        total_yes = 0
        count_yes = 0
        for row in row_range:
            if values[row][col] is not '?':
                if values[row][10] == 'Yes':
                    total_yes += float(values[row][col])
                    count_yes += 1
                else:
                    total_no += float(values[row][col])
                    count_no += 1
        new_yes = round((total_yes / count_yes), 5)
        new_no = round((total_no / count_no), 5)
        new_both = round((total_yes + total_no) / (count_yes + count_no), 5)
        for row in row_range:
            if values[row][col] is '?':
                if conditional:
                    if values[row][10] == 'Yes':
                        df.iat[row, col] = new_yes
                    else:
                        df.iat[row, col] = new_no
                else:
                    df.iat[row, col] = new_both
    df.to_csv(filename)
    return df


def mae(original, df_imputed, df_dataset_complete):
    col_range = range(original.shape[1])
    row_range = range(original.shape[0])
    original_values = original.to_numpy()
    imputed_values = df_imputed.to_numpy()
    complete_values = df_dataset_complete.to_numpy()
    count = 0
    total = 0
    for col in col_range:
        for row in row_range:
            if original_values[row][col] is '?':
                count += 1
                total += abs(float(imputed_values[row][col]) - float(complete_values[row][col]))

    return round(total / count, 4)


if __name__ == "__main__":
     main()

