Chạy file :
- chạy file video có sẵn : python3 demo.py -v path-to-video-file
 trong đó : path-to-video-file là đường dẫn tới file video muốn chạy
- chạy bằng camera : python3 demp.py -c 0
 có thể thay 0 bằng các số tương ứng cổng camera kết nối

Khả năng nhận diện:
    Một người có thể sinh ra nhiều folder trong thư mục temp
    Khi đưa vào face_db thì cần có đủ góc mặt và chọn ra những mặt tốt sẽ cho khả năng nhận diện tốt hơn
    Trong phiên bản này một người khi đã có trong face_db thì khi người đấy vào ngoài việc nhận diện thì
    vẫn có lưu mặt trong thư mục temp để làm giàu dữ liệu.

Đề nghị:
    Do hiện tại camera đặt ở góc nghiêng so với đường đi nên số lượng mặt tốt thu được rất ít
    ảnh hưởng xấu tới khả năng nhận diện. 
    Bởi vì để thu thập được dữ liệu từ góc camera này phải chấp nhận cả những mặt xấu.
    Nên để camera ở góc chính diện hơn.