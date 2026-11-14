# Định dạng dữ liệu Day 7

Dữ liệu được chuẩn bị từ văn bản Bộ luật Lao động Việt Nam năm 2019 (Luật số 45/2019/QH14):

- `chunks.json`: Tập hợp 436 chunks văn bản pháp lý kế thừa từ Strategy B của Day 6. Mỗi chunk bao gồm: `id`, `source_id`, `source`, `page`, `text`, `metadata` (`chapter`, `strategy`).
- `questions.json`: Bộ 20 câu hỏi có đáp án (`15 dev` + `5 test`) phục vụ đánh giá retrieval độc lập qua `baseline.py`. Mỗi câu có `id`, `split`, `question`, `expected_answer`, `gold` (chuỗi ký tự nguyên văn *verbatim substring* khớp với chunk).
- `cases.json`: Bộ 24 câu hỏi benchmark phục vụ đánh giá Agent (`17 dev` + `7 test`). Bao gồm 20 câu có đáp án và 4 câu thiếu bằng chứng ngoài phạm vi (`q_unans_dev_1`, `q_unans_dev_2`, `q_unans_test_1`, `q_unans_test_2`) để kiểm thử năng lực từ chối (*abstention*).
- `example_*.json`: Các tệp dữ liệu mẫu ban đầu từ ban tổ chức.
