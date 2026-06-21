# Mô tả thư mục `adult`

Thư mục `adult` chứa bộ dữ liệu Adult (Census Income) được sử dụng phổ biến cho các bài toán phân loại thu nhập.

## Nội dung chính

- `adult.data`
  - Tập dữ liệu huấn luyện gốc.
  - Dữ liệu dạng CSV không có tiêu đề cột.
  - Mỗi dòng là một cá nhân, với thuộc tính đầu ra là nhãn `>50K` hoặc `<=50K`.

- `adult.test`
  - Tập dữ liệu kiểm tra gốc.
  - Định dạng tương tự `adult.data` nhưng dùng để đánh giá mô hình.

- `adult.names`
  - Mô tả chi tiết bộ dữ liệu.
  - Bao gồm nguồn dữ liệu, mục tiêu dự đoán và các thuộc tính.
  - Nêu ra 14 thuộc tính đầu vào và 1 lớp đích.

- `old.adult.names`
  - Phiên bản cũ của file mô tả.
  - Cung cấp thêm thông tin tham khảo, kết quả thử nghiệm và thống kê phân phối lớp.

- `Index`
  - File chỉ mục liệt kê các file trong thư mục kèm kích thước và ngày tạo.

## Thông tin dữ liệu

- Nguồn: Bộ dữ liệu Adult từ Cục Điều tra Dân số Hoa Kỳ (1994 Census).
- Mục tiêu: Dự đoán liệu một người có thu nhập trên 50K đô la một năm hay không.
- Số lượng bản ghi: 48.842 (tổng), trong đó train ~32.561, test ~16.281.
- Thuộc tính:
  1. `age` (liên tục)
  2. `workclass` (danh mục)
  3. `fnlwgt` (liên tục)
  4. `education` (danh mục)
  5. `education-num` (liên tục)
  6. `marital-status` (danh mục)
  7. `occupation` (danh mục)
  8. `relationship` (danh mục)
  9. `race` (danh mục)
  10. `sex` (danh mục)
  11. `capital-gain` (liên tục)
  12. `capital-loss` (liên tục)
  13. `hours-per-week` (liên tục)
  14. `native-country` (danh mục)
  15. `class` (đích: `>50K`, `<=50K`)

## Ghi chú

- Dữ liệu có giá trị thiếu được biểu diễn bằng `?`.
- Tập dữ liệu này thường được dùng cho các thuật toán học máy như cây quyết định, Naive Bayes, và logistic regression.
