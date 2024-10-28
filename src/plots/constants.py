from pathlib import Path


def ext_path(p):
    return "results/ray/" + p + "/evalresults_best_test.json"


# main agent paths
main_agent_label_dir_pairs = [
    ("Min",  ext_path("2024_06_02_11_48_17_eval-only-min")),
    ("Max",  ext_path("2024_06_02_11_49_15_eval-only-max")),
    ("Random",  ext_path("2024_06_02_11_49_55_eval-only-random")),
    ("Greedy-8",  ext_path("2024_06_02_11_50_47_eval-only-greedy-8")),
    ("Greedy-8-Minerva",  ext_path("2024_06_02_12_03_10_eval-only-greedy-8-minerva")),
    # ("Greedy-5", "results/ray/2024_05_26_16_19_19_eval-only-greedy-5/evalresults_best_test.json"),
    # ("Greedy-5-Minerva", "results/ray/2024_05_26_17_00_17_eval-only-greedy-5-minerva/evalresults_best_test.json"),
    # ("PPO",  ext_path("2024_05_27_09_14_47_ppo-500-lr1e-5-nosharing")),
    ("PPO", ext_path("2024_06_06_14_32_45_ppo-500-lr1e-5-nosharing-fs8-2")),
    ("PPO-Minerva",  ext_path("2024_06_06_11_16_45_ppo-500-lr1e-5-nosharing-minerva-1")),
    # ("Min", wrap_path("2024_05_29_09_25_29_eval-only-min")),
    # ("Min-Minerva", wrap_path("2024_05_29_09_40_20_eval-only-min-minerva")),
    # ("Max", wrap_path("2024_05_29_09_27_20_eval-only-max")),
    # ("Max-Minerva", wrap_path("2024_05_29_09_42_53_eval-only-max-minerva")),
    # ("Random", wrap_path("2024_05_29_09_28_46_eval-only-random")),
    # ("Random-Minerva", wrap_path("2024_05_29_09_44_42_eval-only-random-minerva")),
]


ppo_ablation_label_dirs_pairs = [
    # IMPORTANT: order all runs by final result (as seen in the plot)
    ("PPO", [
        "results/ray/2024_06_06_14_32_45_ppo-500-lr1e-5-nosharing-fs8-2/PPO_StreamingEnv_a3ded_00000_0_2024-06-06_14-32-45",
        "results/ray/2024_06_05_16_51_57_ppo-500-lr1e-5-nosharing-fs8-1/PPO_StreamingEnv_eb778_00000_0_2024-06-05_16-51-57",
        "results/ray/2024_06_04_19_14_04_ppo-500-lr1e-5-nosharing-fs8-0/PPO_StreamingEnv_9c057_00000_0_2024-06-04_19-14-04",
    ]),
    # ("PPO-050", [
    #     "results/ray/2024_06_11_17_55_57_ppo-500-lr1e-5-nosharing-fs8-qf0.5-0/PPO_StreamingEnv_db40a_00000_0_2024-06-11_17-55-57",
    #     "results/ray/2024_06_12_01_40_38_ppo-500-lr1e-5-nosharing-fs8-qf0.5-1/PPO_StreamingEnv_c52f1_00000_0_2024-06-12_01-40-38",
    #     "results/ray/2024_06_12_09_27_15_ppo-500-lr1e-5-nosharing-fs8-qf0.5-2/PPO_StreamingEnv_f4d44_00000_0_2024-06-12_09-27-15",
    # ]),
    # ("PPO-075", [
    #     "results/ray/2024_06_12_05_33_22_ppo-500-lr1e-5-nosharing-fs8-qf0.75-1/PPO_StreamingEnv_48ae8_00000_0_2024-06-12_05-33-22",
    #     "results/ray/2024_06_11_21_48_10_ppo-500-lr1e-5-nosharing-fs8-qf0.75-0/PPO_StreamingEnv_4ba05_00000_0_2024-06-11_21-48-10",
    #     "results/ray/2024_06_12_13_21_09_ppo-500-lr1e-5-nosharing-fs8-qf0.75-2/PPO_StreamingEnv_a1ca4_00000_0_2024-06-12_13-21-09",
    # ]),
    ("PPO-Sharing", [
        "results/ray/2024_06_05_20_44_52_ppo-500-lr1e-5-fs8-1/PPO_StreamingEnv_759e8_00000_0_2024-06-05_20-44-52",
        "results/ray/2024_06_04_23_04_29_ppo-500-lr1e-5-fs8-0/PPO_StreamingEnv_cbe34_00000_0_2024-06-04_23-04-29",
        "results/ray/2024_06_06_18_26_11_ppo-500-lr1e-5-fs8-2/PPO_StreamingEnv_3ffde_00000_0_2024-06-06_18-26-11",
    ]),
    ("PPO-LSTM", [
        "results/ray/2024_06_06_00_41_33_ppo-500-lr1e-5-nosharing-lstm-1/PPO_StreamingEnv_85c55_00000_0_2024-06-06_00-41-33",
        "results/ray/2024_06_05_02_58_59_ppo-500-lr1e-5-nosharing-lstm-0/PPO_StreamingEnv_8eab3_00000_0_2024-06-05_02-58-59",
        "results/ray/2024_06_06_22_22_12_ppo-500-lr1e-5-nosharing-lstm-2/PPO_StreamingEnv_38d86_00000_0_2024-06-06_22-22-12",
    ]),
    ("PPO-Minerva", [
        "results/ray/2024_06_06_11_16_45_ppo-500-lr1e-5-nosharing-minerva-1/PPO_StreamingEnv_42655_00000_0_2024-06-06_11-16-45",
        "results/ray/2024_06_05_13_38_39_ppo-500-lr1e-5-nosharing-minerva-0/PPO_StreamingEnv_ead50_00000_0_2024-06-05_13-38-39",
        "results/ray/2024_06_07_08_58_56_ppo-500-lr1e-5-nosharing-minerva-2/PPO_StreamingEnv_2c2e1_00000_0_2024-06-07_08-58-56",
    ]),
]
assert ppo_ablation_label_dirs_pairs[0][0] == "PPO"
best_ppo_agent_dir = ppo_ablation_label_dirs_pairs[0][1][0]
worst_ppo_agent_dir = ppo_ablation_label_dirs_pairs[0][1][-1]

ppo_ablation_coeff_path_dict = {
    # 'b': ext_path("2024_06_02_12_03_10_eval-only-greedy-8-minerva"),
    0.25: ext_path("2024_06_06_14_32_45_ppo-500-lr1e-5-nosharing-fs8-2"),
    0.5: ext_path("2024_06_11_17_55_57_ppo-500-lr1e-5-nosharing-fs8-qf0.5-0"),
    0.75: ext_path("2024_06_12_05_33_22_ppo-500-lr1e-5-nosharing-fs8-qf0.75-1"),
}

plots_dir = Path("results/plots/")
plots_dir.mkdir(exist_ok=True, parents=True)
