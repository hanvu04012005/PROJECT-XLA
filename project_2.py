import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import math
import scipy.fftpack
import pandas as pd

# Load model đã train
model = YOLO("runs/detect/train4/weights/best.pt")

def Butterworth_Highpass_Filter(im_1):
    c = scipy.fftpack.fft2(im_1)
    d = scipy.fftpack.fftshift(c)
    M, N = d.shape
    H = np.ones((M, N))
    center1 = M / 2
    center2 = N / 2
    d_0 = 30.0  # bán kính cắt
    t1 = 1  # bậc của bộ lọc

    for i in range(M):
        for j in range(N):
            r = math.sqrt((i - center1) ** 2 + (j - center2) ** 2)
            H[i, j] = 1 - 1 / (1 + (r / d_0) ** (2 * t1))

    con = d * H
    e = abs(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(con)))
    e = (e - np.min(e)) / (np.max(e) - np.min(e)) * 255
    return e.astype(np.uint8)

def Butterworth_Highpass_Color(image):
    img_cv = np.array(image)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l_filtered = butterworth_filter(l)
    l_sharpened = cv2.addWeighted(l, 1.0, l_filtered, 1.0, 0)  # tăng rõ
    merged = cv2.merge((l_sharpened, a, b))
    rgb_result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_result)

def butterworth_filter(im_1, d_0=30.0, t1=1):
    M, N = im_1.shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H = 1 - 1 / (1 + (D / d_0) ** (2 * t1))

    F = scipy.fftpack.fftshift(scipy.fftpack.fft2(im_1))
    G = H * F
    g = np.abs(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(G)))
    g = np.clip((g - np.min(g)) / (np.max(g) - np.min(g)) * 255, 0, 255)
    return g.astype(np.uint8)

# Đọc dữ liệu từ file Excel chỉ 1 lần
df_dinhduong = pd.read_excel("DinhDuong.xlsx")
df_congdung = pd.read_excel("CongDung.xlsx")
def lay_thong_tin_excel(ten, loai):
    ten = ten.strip().lower()
    try:
        if loai == "Công dụng":
            for _, row in df_congdung.iterrows():
                if row["Quả"].strip().lower() in ten:
                    return row["Công dụng"]
        elif loai == "Dinh dưỡng":
            for _, row in df_dinhduong.iterrows():
                if row["Quả"].strip().lower() in ten:
                    return row["Dinh dưỡng"]
    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return [], []



def hien_thong_tin_excel(ten, mode, loai_thong_tin):
    if mode == "lam_ro_doi_tuong" and ten.strip():
        return gr.update(value=lay_thong_tin_excel(ten, loai_thong_tin), visible=True)
    else:
        return gr.update(visible=False)


# Hàm xử lý ảnh
def xu_ly(images, mode, loai_trai_cay, butter_option, ten):

    if not images:
        return [], []

    all_images = []
    counts = {}
    for img_file in images:
        original_img = Image.open(img_file.name).convert("RGB")

        names = model.names
        results = model.predict(original_img, conf=0.25, device="mps")[0]

        try:
            font = ImageFont.truetype("arial.ttf", size=32)
        except:
            font = ImageFont.truetype("DejaVuSans.ttf", size=32)

        # === CHẾ ĐỘ 1: Làm rõ đối tượng ===
        if mode == "lam_ro_doi_tuong":
            blurred = original_img.filter(ImageFilter.GaussianBlur(radius=5))
            img = blurred.copy()
            draw = ImageDraw.Draw(img)
        # === CHẾ ĐỘ 2: Làm rõ ảnh toàn bộ bằng CLAHE ===
        elif mode == "lam_ro_anh":
            # Chuyển PIL -> OpenCV để xử lý CLAHE
            img_cv = np.array(original_img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            # CLAHE trên kênh L trong LAB
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(enhanced_rgb)
            draw = ImageDraw.Draw(img)
        elif mode == "butterworth":
            if butter_option == "Chuyển ảnh xám":
                gray = original_img.convert("L")
                filtered = Butterworth_Highpass_Filter(np.array(gray))
                combined = cv2.addWeighted(np.array(gray), 0.7, filtered, 1.2, 0)
                img = Image.fromarray(combined).convert("RGB")

            else:
                img = Butterworth_Highpass_Color(original_img)
            draw = ImageDraw.Draw(img)


        # === Trường hợp không rõ ===
        else:
            img = original_img.copy()
            draw = ImageDraw.Draw(img)

        # Duyệt qua kết quả nhận diện
        for box in results.boxes:
            cls_id = int(box.cls.item())
            name = names[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            counts[name] = counts.get(name, 0) + 1

            if mode == "lam_ro_doi_tuong" :
                if loai_trai_cay.lower() in name.lower():
                    roi = original_img.crop((x1, y1, x2, y2))
                    roi = roi.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                    img.paste(roi, (x1, y1))
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    draw.text((x1, max(y1 - 35, 0)), f"{name} {conf:.2f}", fill="green", font=font)
            else:
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, max(y1 - 35, 0)), f"{name} {conf:.2f}", fill="green", font=font)
            
            
        all_images.append(img)

    bang = [[k, v] for k, v in counts.items()]
    return all_images, bang

# Giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🍎 Ứng dụng Nhận Diện & Phân Loại Trái Cây")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.Markdown("### 📁 Tải lên ảnh")
            images_input = gr.Files(label="Chọn 1 hoặc nhiều ảnh", file_types=["image"], file_count="multiple")
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Kết quả phân loại")
            nut_phan_loai = gr.Button("Phân loại", variant="primary")
            bang_phan_loai = gr.Dataframe(headers=["Tên", "Số lượng"], interactive=False)

    gr.Markdown("### ⚙️ Chọn chế độ xử lý")
    with gr.Row():
        with gr.Column():
            radio_mode = gr.Radio(
                choices=[
                    ("Làm rõ ảnh lớn", "lam_ro_anh"),
                    ("Làm rõ đối tượng", "lam_ro_doi_tuong"),
                    ("Bộ lọc thông cao Butterworth", "butterworth")
                ],
                label="Chế độ", value=None, interactive=True
            )

        with gr.Column():
            ten_loai = gr.Textbox(label="Nhập tên trái cây cần làm rõ đối tượng", visible=False)
            butter_mode = gr.Radio(
                choices=["Giữ màu gốc", "Chuyển ảnh xám"],
                label="Chế độ màu cho Butterworth",
                value="Giữ màu gốc",
                visible=False
            )

    nut_enter = gr.Button("Bắt đầu xử lý")

    gr.Markdown("### Kết quả xử lý ")
    with gr.Row():
        with gr.Column(scale=3):
            ket_qua_anh = gr.Gallery(
                label="Ảnh sau xử lý",
                show_label=False,
                columns="auto",
                object_fit="contain",
                height="auto",
                preview=True
            )
        with gr.Column(scale=2):
            chon_thong_tin = gr.Radio(
                choices=["Công dụng", "Dinh dưỡng"],
                value="Công dụng",
                label="Chọn thông tin cần hiển thị",
            )
            thong_tin_output = gr.Textbox(
                label="Thông tin chi tiết",
                interactive=False,
                visible=False,
                lines=10,
            )

    # ================= Sự kiện ==================
    def hien_textbox(mode):
        return (
            gr.update(visible=(mode == "lam_ro_doi_tuong")),
            gr.update(visible=(mode == "butterworth"))
        )

    radio_mode.change(hien_textbox, inputs=radio_mode, outputs=[ten_loai, butter_mode])
    ten_loai.change(fn=hien_thong_tin_excel, inputs=[ten_loai, radio_mode, chon_thong_tin], outputs=thong_tin_output)
    chon_thong_tin.change(fn=hien_thong_tin_excel, inputs=[ten_loai, radio_mode, chon_thong_tin], outputs=thong_tin_output)
    nut_enter.click(fn=xu_ly, inputs=[images_input, radio_mode, ten_loai, butter_mode], outputs=[ket_qua_anh, bang_phan_loai])
    nut_phan_loai.click(fn=xu_ly, inputs=[images_input, radio_mode, ten_loai, butter_mode], outputs=[ket_qua_anh, bang_phan_loai])

# Chạy ứng dụng
demo.launch(share=True)
gr.close_all()