# Nhận diện biển số bằng OpenCV

## Chương trình xử lý được chia làm 3 file chính đó là:
  - Preprocess.py : Bước tiền xử lý ảnh 
  - FindChar.py : Xử lý các ký tự (Contours)
  - FindPlate.py : Trích xuất các biển từ các ký tự tìm được
  
  Chương trình sử dụng thuật toán KNN(K-Nearest Neighbor) bài toán được đưa ra ở đây là phân loại(Classification). Vậy nên chúng ta có 2 file dữ liệu đó là classfication.txt và flattened_image.txt
  
## Dữ liệu dùng để trainning:
  ### Dữ liệu dùng để trainning bao gồm 2 file:
  #### - classfication.txt : là file lable với các ký tự biểu diễn dướng dạng mã ASCI
  
  ![alt text](/image/Capture.PNG)
  
  #### - flattened_image.txt : là các file hình ảnh của ký tự đưởng biểu diễn dưới dạng ma trận
  
  ![alt_text](/image/Capture2.PNG)
  
  Mỗi hình ảnh ký tự đó có kích thước là 30x20 pixel . Đây là hình của một ký tự sau khi được 
  chuyển từ ma trận sang hình ảnh
  
  ![alt_text](/image/Capture3.PNG)

## Quá trình xử lý của chương trình 
