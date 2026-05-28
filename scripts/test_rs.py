import sys
import math

def load_data(filename):
    """파일을 읽어서 {(user_id, item_id): rating} 형태의 딕셔너리로 반환"""
    data_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            u, i, r = int(parts[0]), int(parts[1]), float(parts[2])
            data_dict[(u, i)] = r
    return data_dict

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_rs.py [test_file] [prediction_file]")
        sys.exit(1)

    test_file = sys.argv[1]       # 정답이 있는 원본 테스트 파일 (예: u1.test)
    pred_file = sys.argv[2]       # 우리가 만든 예측 파일 (예: u1.base_prediction.txt)

    print(f"Loading Ground Truth: {test_file}")
    truth_data = load_data(test_file)
    
    print(f"Loading Predictions: {pred_file}")
    pred_data = load_data(pred_file)

    squared_error_sum = 0.0
    count = 0

    # 정답 파일에 있는 유저-아이템 쌍을 기준으로 예측값과 비교
    for (u, i), actual_rating in truth_data.items():
        if (u, i) in pred_data:
            predicted_rating = pred_data[(u, i)]
            squared_error_sum += (actual_rating - predicted_rating) ** 2
            count += 1
        else:
            print(f"Warning: Missing prediction for User {u}, Item {i}")

    if count == 0:
        print("Error: No matching user-item pairs found between the two files.")
        sys.exit(1)

    # RMSE 계산 공식 적용
    rmse = math.sqrt(squared_error_sum / count)
    
    print("-" * 30)
    print(f"Total evaluated pairs: {count}")
    print(f"🔥 FINAL RMSE: {rmse:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    main()