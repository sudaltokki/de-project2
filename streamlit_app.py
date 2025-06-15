import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import torchvision.models as models

# ==================== 설정 ====================
# 모델 로드
yolo_model = YOLO('first_model_best.pt')  # 재질 분류용
# 속성 분류 모델 (비워도 작동하도록 처리)
cnn_model = None  # 추후 torch.load('cnn_model.pth') 가능

# 속성 클래스
attribute_names = ['broken', 'contaminated', 'contains_liquid',
                   'has_pattern', 'has_plastic_cap', 'has_metal_cap',
                   'has_plastic_label']

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Rule 기반 재활용 판단
def predict_rule(material, attributes):
    if material == 'plastic_pet' and 'contaminated' in attributes:
        return "재활용 불가", "오염된 페트병은 재활용이 어려움"
    if material == 'glass' and 'broken' not in attributes:
        return "재활용 가능", "깨끗하고 손상되지 않은 유리는 재활용 가능"
    return "재활용 불가", "재질 또는 속성 조건 불충분"


# ==================== UI 시작 ====================
st.set_page_config(page_title="재활용 여부 판단 AI", layout="centered")
st.title("♻️ 재활용 여부 판단 AI")

# 사이드바 메뉴
option = st.sidebar.selectbox("입력 방식 선택", ("이미지 업로드", "카메라 촬영"))

image = None

# 이미지 업로드 방식
if option == "이미지 업로드":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# 카메라 촬영 방식
elif option == "카메라 촬영":
    camera_image = st.camera_input("카메라로 사진을 찍어주세요")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# ==================== 예측 수행 ====================

if image:
    st.image(image, caption="입력 이미지", use_container_width=True)

    img_array = np.array(image)
    results = yolo_model(img_array)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy()

    results_list = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))
        input_tensor = transform(cropped).unsqueeze(0)

        # 속성 예측 (비워도 실행됨)
        if cnn_model:
            with torch.no_grad():
                attr_logits = cnn_model(input_tensor)
                attr_preds = torch.sigmoid(attr_logits).squeeze() > 0.5
                attr_labels = [attr for idx, attr in enumerate(attribute_names) if attr_preds[idx]]
        else:
            attr_labels = []  # 속성 예측 모델 없을 경우

        material_class = results[0].names[int(cls_ids[i])]
        decision, reason = predict_rule(material_class, attr_labels)
        results_list.append((cropped, material_class, attr_labels, decision, reason))

    st.subheader("🔍 분류 결과")

    for idx, (img, material, attrs, decision, reason) in enumerate(results_list):
        with st.expander(f"📦 객체 {idx+1} - {'✅' if '가능' in decision else '❌'} {decision}", expanded=False):
            st.image(img, caption=f"객체 {idx+1}", width=200)
            st.markdown(f"**재질:** {material}")
            st.markdown(f"**속성:** {', '.join(attrs) if attrs else '없음'}")
            st.markdown(f"**판단 결과:** {'✅' if '가능' in decision else '❌'} {decision}")
            st.markdown(f"**사유:** {reason}")
