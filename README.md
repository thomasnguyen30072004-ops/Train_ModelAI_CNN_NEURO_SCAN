Cài 2 thư viện:

pip install streamlit tensorflow opencv-python matplotlib

pip install imutils

RUN APP: 

open termial trong VScode: streamlit run app.py

Giao diện web hỗ trợ chuyển đổi giữa 2 model PRO or FINAL để test

Pro được nâng cấp từ FINAL:

-> Loại bỏ viền trắng xương sọ ngay từ cửa vào để AI không bị lừa

-> Ảnh to hơn -----> Nhìn chi tiết rõ hơn

-> Thay lớp Flatten() bằng GlobalAveragePooling2D() giúp giảm số lượng tham số khổng lồ, làm model nhẹ hơn mà vẫn thông minh.

từ khóa tìm kiếm ảnh test:

Brain MRI T2 weighted || Brain MRI FLAIR || Brain MRI Axial plane
