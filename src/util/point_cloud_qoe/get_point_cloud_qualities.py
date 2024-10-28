import pandas as pd

ratings = pd.read_csv("ratings.csv")

vpcc_ratings = ratings.loc[ratings['encode_method'] == 'V-PCC'].reset_index()
# mean over all objects, all participants, all distances
vpcc_qoe = vpcc_ratings.groupby(["frame_rate", "quantization_level_index"]).mean(numeric_only=True)["qoe"].to_frame()
print(vpcc_qoe)

resources = pd.read_csv("resources.csv")
vpcc_resources = resources.loc[resources['encode_method'] == 'V-PCC'].reset_index()
# mean over all objects
vpcc_bitrate = vpcc_resources.groupby(["object_rate", "quantization_level_index"]).mean(numeric_only=True)["compressed_bit_rate"].to_frame()
# rename index for join
vpcc_bitrate.index.rename(["frame_rate", "quantization_level_index"], inplace=True)
print(vpcc_bitrate)

# combine qualities and bitrates
vpcc = vpcc_qoe.join(vpcc_bitrate)
print(vpcc)

best_vpcc = vpcc.sort_values("qoe", ascending=False).reset_index(drop=False)
# then filter out all rows for which one can achieve a better qoe with a lower bitrate
for i in reversed(range(len(best_vpcc))):
    row = best_vpcc.iloc[i]
    less_br_rows = best_vpcc[(best_vpcc["compressed_bit_rate"] < row["compressed_bit_rate"])]
    if less_br_rows["qoe"].max() >= row["qoe"]:
        best_vpcc.drop(i, axis=0, inplace=True)

print(best_vpcc)
best_vpcc.to_csv("best_vpcc.csv", index=False)
