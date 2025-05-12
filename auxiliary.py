import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid

def cumsum_plot(results_df,plot_text):
    """
    This computes the daily MSE, then does a cumulative sum 
    and plots the average cumsum over time.x
    """

    # ÇOK ÖNEMLİ: convert from ds to horizon
    # 1) Önce mutlaka zaman sırasına göre sıralayın
    results_df = results_df.sort_values(['unique_id', 'ds'])

    # 2) Grup içi sıralamaya karşılık gelen bir sayaç ekleyin
    results_df['horizon'] = results_df.groupby('unique_id').cumcount() + 1

    model_names = [
        "lgb",
        "bsdt",
        "bsdtq"
    ]

    plt.figure(0)
    # plt.figure(1, figsize=(8, 5))
    # plt.figure(2, figsize=(8, 5))

    for model_name in model_names:
        # print(model_name)
        # Squared errors
        results_df[f'{model_name}_sq_err'] = (
            results_df['y'] - results_df[model_name])**2
        results_df[f'{model_name}_smape'] = abs(
            results_df['y'] - results_df[model_name]) / (abs(results_df['y']) + abs(results_df[model_name]))

        # Group by day, average
        daily_mse = results_df.groupby(
            'horizon')[f'{model_name}_sq_err'].mean().sort_index()
        daily_smape = results_df.groupby(
            'horizon')[f'{model_name}_smape'].mean().sort_index()

        # Cumulative sum
        mse_cumsum = daily_mse.cumsum()
        smape_cumsum = daily_smape.cumsum()

        # Average cumsum up to day i
        n = np.arange(1, len(mse_cumsum)+1)
        cumavg_mse = mse_cumsum / n
        cumavg_smape = smape_cumsum / n

        #plt.figure(0)
        plt.plot(n, cumavg_mse, label=model_name)

        # plt.figure(1,figsize=(8,5))
        # plt.plot(n, daily_mse, label=model_name)

    

    #plt.figure(0, figsize=(8, 5))
    plt.title("$cMSE_t$\n (All series averaged)")
    plt.xlabel("t")
    plt.ylabel("MSE")
    # plt.xlim([1, 48])
    # plt.ylim([0,5e6])
    # plt.xticks([12,24,36,48])
    plt.grid(True)
    plt.legend()

    plt.text(103,0,plot_text,fontsize = 7)

    plt.savefig(uuid.uuid4().hex + ".pdf", bbox_inches="tight")
    plt.close()