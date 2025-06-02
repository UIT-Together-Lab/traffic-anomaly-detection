import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

def read_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        data = [float(value.strip()) for value in content.split(',') if value.strip()]
    return np.array(data)

def analyze_predictions(predictions):
    unique_values = np.unique(predictions)
    
    print(f"Giá trị duy nhất trong dữ liệu: {unique_values}")
    
    if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
        print("Nhãn nhị phân (0 và 1)")
        anomaly_count = np.sum(predictions)
    else:
        print("Không xác định được loại nhãn hoặc dữ liệu không phải nhị phân")
        return
    
    print(f"Tổng số frame: {len(predictions)}")
    print(f"Số frame bất thường: {anomaly_count}")
    print(f"Tỷ lệ frame bất thường: {anomaly_count / len(predictions):.2%}")


def analyze_threshold(predictions, true_labels):
    # # Vẽ histogram
    plt.figure(figsize=(10, 5))
    plt.hist(predictions, bins=50)
    plt.title("Phân phối điểm số dự đoán")
    plt.xlabel("Điểm số")
    plt.ylabel("Số lượng")
    plt.show()

    # Tính precision, recall, và ngưỡng
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    
    # Tính F1-score cho mỗi ngưỡng
    f1_scores = [f1_score(true_labels, predictions >= threshold) for threshold in thresholds]
    
    # Tìm ngưỡng tối ưu dựa trên F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Vẽ đường cong Precision-Recall
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', label=f'Optimal threshold: {optimal_threshold:.2f}')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    print(f"Ngưỡng tối ưu dựa trên F1-score: {optimal_threshold:.4f}")
    print(f"F1-score tại ngưỡng tối ưu: {f1_scores[optimal_idx]:.4f}")

    # Thêm phân tích bổ sung
    print(f"\nPhân phối dự đoán:")
    print(f"Giá trị nhỏ nhất: {predictions.min():.4f}")
    print(f"Giá trị lớn nhất: {predictions.max():.4f}")
    print(f"Giá trị trung bình: {predictions.mean():.4f}")
    print(f"Độ lệch chuẩn: {predictions.std():.4f}")

    # Phân tích với ngưỡng tối ưu
    predictions_binary = predictions >= optimal_threshold
    true_positives = np.sum((predictions_binary == 1) & (true_labels == 1))
    false_positives = np.sum((predictions_binary == 1) & (true_labels == 0))
    true_negatives = np.sum((predictions_binary == 0) & (true_labels == 0))
    false_negatives = np.sum((predictions_binary == 0) & (true_labels == 1))

    print(f"\nKết quả với ngưỡng tối ưu ({optimal_threshold:.4f}):")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")


if __name__ == '__main__':
    file_path = "./save_label_final_UIT_DroneAnomaly_Foggy_Bike_Roundabout.txt"
    true_labels_path = "/storageStudents/ncsmmlab/tungufm/Vehicle Roundabout/sequence2/test/test_frame_masks"  

    predictions = read_data(file_path)

    true_labels = read_data(true_labels_path)
    true_labels = (true_labels > 0.5).astype(int)

    analyze_threshold(predictions, true_labels)  

    #predictions = read_data(file_path)
    #analyze_predictions(predictions)

