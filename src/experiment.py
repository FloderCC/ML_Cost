import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from src.utils.energy_simulator import simulate_energy_consumption
from src.utils.model_utils import reset_setup, create_models
from src.utils.resources_monitor import monitor_tic, monitor_toc
from src.utils.dataset_utils import *

# keras.utils.disable_interactive_logging()

dataset_setup_list = [
    ['KPI-KQI', [], 'Service'],  # 165 x 14
    ['UNAC', ['file'], 'output'],  # 389 x 23
    ['IoT-APD', ['second'], 'label'],  # 10845 x 17
    ['QOE_prediction_ICC2018', ['RebufferingRatio', 'AvgVideoBitRate', 'AvgVideoQualityVariation'], 'StallLabel'],  # 69129 x 51
    ['RT_IOT2022', ['no'], 'Attack_type'],  # 123117, 85
    ['5G_Slicing', [], 'Slice Type (Output)'],  # 466739, 9

    ['IoT-DNL', [], 'normality'],  # 477426 x 14
    ['X-IIoTID', ['Date', 'Timestamp', 'class1', 'class2'], 'class3'],  # 820834 x 68
    ['IoTID20', ['Flow_ID', 'Cat', 'Sub_Cat'], 'Label'],  # 625783 x 86
    ['DDOS-ANL', [], 'PKT_CLASS'],  # 2160668 x 28

    # backup
    # ['BoTNeTIoT-L01', ['Device_Name', 'Attack', 'Attack_subType'], 'label'],  # 7062606 x 27 SIGKILL
    # Unknown: https://www.kaggle.com/datasets/puspakmeher/networkslicing

]

seeds = [5, 7, 42, 11]

# count
setup_number = 1
setup_quantity = len(dataset_setup_list) * len(create_models(42)) * len(seeds)

# results
results_header = ['Dataset',
                  'Class Imbalance Ratio',
                  'Gini Impurity',
                  'Entropy',
                  'Number of samples',
                  'Number of features',
                  'Completeness',
                  'Consistency',
                  'Uniqueness',
                  'Redundancy (avg)',
                  'Redundancy (std)',
                  'Max value',
                  'Min value',
                  'Global avg',
                  'Global std',
                  'Avg of features\' avg',
                  'Std of features\' avg',
                  'Avg of features\' std',
                  'Std of features\' std',
                  'Seed',
                  'Model',
                  'TR TIME', 'TR CPU%', 'TR Energy (J)',
                  'TE TIME', 'TE CPU%', 'TE Energy (J)',
                  'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC'
                  ]

all_results = []

for dataset_setup in dataset_setup_list:
    dataset_name = dataset_setup[0]
    useful_columns = dataset_setup[1]
    class_name = dataset_setup[2]
    result_from_dataset = [dataset_name]

    # loading the dataset
    dataset_folder = f"../datasets/{dataset_name}"
    df = pd.read_csv(f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}", low_memory=False)

    print(f"\n------- Started execution with dataset {dataset_name} {df.shape} -------")
    # removing not useful columns
    if len(useful_columns) > 0:
        print(f"Removing columns {useful_columns}")
        df = df.drop(columns=useful_columns)

    # describe the raw dataset
    print("Describing raw dataset ...")
    result_from_dataset += describe_raw_dataset(df, class_name)

    # codify & prepare the dataset
    print("Codifying & preparing dataset ...")
    df = preprocess_dataset(df)

    # describe the codified dataset
    print("Describing codified dataset ...")
    result_from_dataset += describe_codified_dataset(df, class_name)

    # splitting features & label
    X = df.drop(dataset_setup[2], axis=1)
    y = df[dataset_setup[2]]

    # encoding Y to make it processable with DNN models
    y_encoded = pd.get_dummies(y)

    for seed in seeds:
        reset_setup(seed)
        models = create_models(seed)

        # splitting the dataset in train and test
        x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2,
                                                                            random_state=seed, stratify=y_encoded)

        # parsing y_test to a multiclass target
        y_test = y_test_encoded.idxmax(axis=1)

        print(f"Executing ML models")
        for model_name, model in models.items():
            result_from_dataset_and_seed_and_model = result_from_dataset.copy()

            print(f" - {model_name} (Setup {setup_number} / {setup_quantity})")

            is_dnn = model_name.startswith("DNN")
            if is_dnn:
                y_train = y_train_encoded
                # building the tf models in case of DNN
                model.build(x_train.shape[1], y_train.shape[1])
            else:
                # parsing y_train to a multiclass target if the model is not DNN
                y_train = y_train_encoded.idxmax(axis=1)

            # training
            print("   - Training ...")
            monitor_tic()
            model.fit(x_train, y_train)
            tr_action_cpu_percent, tr_action_elapsed_time = monitor_toc()

            # testing
            print("   - Testing ...")
            monitor_tic()
            y_pred = model.predict(x_test)
            inf_action_cpu_percent, inf_action_elapsed_time = monitor_toc()

            if is_dnn:
                y_train = y_train_encoded.idxmax(axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)

            # saving the results
            result_from_dataset_and_seed_and_model += [
                seed,
                model_name,
                tr_action_elapsed_time,
                tr_action_cpu_percent,
                simulate_energy_consumption(tr_action_elapsed_time, tr_action_cpu_percent),
                inf_action_elapsed_time,
                inf_action_cpu_percent,
                simulate_energy_consumption(inf_action_elapsed_time, inf_action_cpu_percent),
                accuracy,
                precision,
                recall,
                f1,
                mcc
            ]

            setup_number += 1

            all_results.append(result_from_dataset_and_seed_and_model)
            # dumping results for a file
            results_df = pd.DataFrame(all_results, index=None, columns=results_header)

            # Write to csv
            results_df.to_csv(f'results/all_results.csv', index=False)

