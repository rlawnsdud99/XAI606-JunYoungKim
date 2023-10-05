# ## 정규화된 데이터 분포 확인
# # Set up the matplotlib figure
# plt.figure(figsize=(16, 6))

# # Plot the distribution of features for training data
# plt.subplot(1, 2, 1)
# sns.boxplot(data=train_data)
# plt.title("Distribution of Normalized Training Features")

# # Plot the distribution of features for validation data
# plt.subplot(1, 2, 2)
# sns.boxplot(data=val_data)
# plt.title("Distribution of Normalized Validation Features")

# plt.tight_layout()
# plt.show()


# # 1. "label" 값 별로 DataFrame을 분리
# unique_labels = val_dataframe["label"].unique()
# results = {}  # 피험자별 validation 결과를 저장할 dictionary

# # 2. 각 피험자별로 validation을 진행
# for label in unique_labels:
#     print(f"iter on subject No.{label}")
#     sub_val_df = val_dataframe[val_dataframe["label"] == label]
#     sub_val_data = (sub_val_df.drop(columns=["label", "class", "time"])).values
#     sub_val_data -= mean
#     sub_val_data /= std
#     sub_val_label = (sub_val_df["class"]).values

#     # SVM 모델로 예측
#     sub_val_predictions = svm_model.predict(sub_val_data)
#     sub_accuracy = accuracy_score(sub_val_label, sub_val_predictions)

#     # 결과 저장
#     results[label] = sub_accuracy
#     print(f"Validation accuracy for label {label}: {sub_accuracy}")

# # 전체 결과 출력
# print("All Validation Results:", results)
