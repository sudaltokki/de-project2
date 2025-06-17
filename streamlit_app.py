import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import os

st.set_page_config(page_title="재활용 여부 판단 AI", layout="centered")

# ==================== 설정 ====================

# 1차 재질 분류 (YOLO)
yolo_model = YOLO('models/first_model_best.pt')

# 2차 세부 재질 분류 (ResNet18)
detailed_materials = ['plastic_pet', 'plastic_hdpe', 'plastic_pp', 'plastic_ps']
resnet18_model = models.resnet18(pretrained=False)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, len(detailed_materials))
resnet18_model.load_state_dict(torch.load("models/second_model_best.pt", map_location=torch.device('cpu')))
resnet18_model.eval()

# 전체 속성 클래스
attribute_names = ['broken', 'contaminated', 'contains_liquid',
                   'has_pattern', 'has_plastic_cap', 'has_metal_cap',
                   'has_plastic_label', 'has_color']

# 3차 속성 분류 (ResNet18 기반)
# 재질별 허용 속성
allowed_attrs_per_material = {
    'aluminum_can': [],
    'glass': ['broken', 'has_color', 'has_metal_cap', 'has_plastic_cap'],
    'iron_can': [],
    'metal': [],
    'paper': ['contaminated'],
    'paper_pack': [],
    'plastic_bottle': ['contaminated', 'has_plastic_cap', 'has_plastic_label'],
    'plastic_pe': ['contaminated', 'has_plastic_cap', 'has_plastic_label'],
    'plastic_pet': ['contaminated', 'has_color', 'has_plastic_cap', 'has_plastic_label'],
    'plastic_pp': ['broken', 'contaminated', 'has_plastic_label'],
    'plastic_ps': ['contaminated', 'has_plastic_label'],
    'styrofoam': ['has_pattern'],
    'vinyl': ['contaminated']
}

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# AttributeClassifier 정의
class AttributeClassifier(nn.Module):
    def __init__(self, num_materials, num_attrs):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.embed = nn.Embedding(num_materials, 64)
        self.fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_attrs),
            nn.Sigmoid()
        )

    def forward(self, x, material_idx):
        x = self.cnn(x)
        m = self.embed(material_idx)
        x = torch.cat([x, m], dim=1)
        return self.fc(x)

# 모든 속성 분류 모델 미리 로딩
@st.cache_resource
def load_all_attribute_models():
    models_dict = {}
    for material, allowed_attrs in allowed_attrs_per_material.items():
        if not allowed_attrs:
            continue
        model_path = f"models/{material}_best.pt"
        if os.path.exists(model_path):
            try:
                model = AttributeClassifier(len(allowed_attrs_per_material), len(allowed_attrs))
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                models_dict[material] = model
                print(f"✅ {material} 모델 로드 완료")
            except Exception as e:
                print(f"⚠️ {material} 모델 로드 실패: {e}")
    return models_dict

attribute_models = load_all_attribute_models()

def run_attribute_classifier(image_tensor, material_name):
    allowed_attrs = allowed_attrs_per_material.get(material_name, [])
    if not allowed_attrs or material_name not in attribute_models:
        return []

    model = attribute_models[material_name]
    material_idx = torch.tensor([list(allowed_attrs_per_material.keys()).index(material_name)])
    with torch.no_grad():
        logits = model(image_tensor, material_idx)
        preds = logits.detach().cpu().numpy().flatten().tolist()
    print(allowed_attrs)
    print(preds)
    return [attr for attr, pred in zip(allowed_attrs, preds) if pred > 0.5]

# ==================== Rule 기반 판단 ====================
def predict_rule(material, attributes):
    reasons = []

    if material.startswith('glass'):
        if 'broken' in attributes:
            reasons.append("깨진 유리는 일반쓰레기로 배출해야 합니다")
        if 'contaminated' in attributes:
            reasons.append("내용물이나 기름 등으로 오염된 유리는 재활용이 어렵습니다")
        if 'contains_liquid' in attributes:
            reasons.append("내용물을 완전히 비운 후 배출해야 합니다")
        if 'has_color' in attributes:
            reasons.append("색깔 있는 유리는 일반쓰레기로 분류됩니다")
        if 'has_metal_cap' in attributes:
            reasons.append("금속 뚜껑은 제거 후 배출해야 합니다")
        if 'has_plastic_cap' in attributes:
            reasons.append("플라스틱 뚜껑은 제거 후 배출해야 합니다")

        if reasons:
            return "재활용 불가", "\n".join(reasons)
        else:
            return "재활용 가능", "깨끗하고 손상되지 않은 유리는 재활용 가능합니다"

    if material.startswith('plastic_bottle'):
        if 'has_plastic_label' in attributes:
            reasons.append("플라스틱 라벨은 제거해야 합니다")
        if 'contaminated' in attributes or 'contains_liquid' in attributes:
            reasons.append("내용물이나 오염물은 씻어낸 후 배출해야 합니다")
        if 'has_metal_cap' in attributes:
            reasons.append("금속 뚜껑은 분리 후 배출해야 합니다")
        if 'has_pattern' in attributes:
            reasons.append("무늬가 있는 병은 스티로폼 등과 혼합되어 재활용이 어렵습니다")

        if reasons:
            return "재활용 불가", "\n".join(reasons)
        else:
            return "재활용 가능", "플라스틱 병은 깨끗하게 배출 시 재활용 가능합니다"

    if material == 'plastic_pet':
        if 'has_plastic_label' in attributes:
            reasons.append("라벨은 분리해야 합니다")
        if 'contaminated' in attributes or 'contains_liquid' in attributes:
            reasons.append("내용물을 비우고 깨끗이 씻은 후 배출해야 합니다")
        if 'has_metal_cap' in attributes:
            reasons.append("금속 캡은 분리해야 합니다")

        return ("재활용 불가", "\n".join(reasons)) if reasons else \
               ("재활용 가능", "색상과 파손 여부 상관없이 재활용 가능합니다")

    if material == 'plastic_pe' or material == 'plastic_pp' or material == 'plastic_ps':
        if 'has_plastic_label' in attributes:
            reasons.append("플라스틱 라벨은 분리 후 배출해야 합니다")
        if 'contaminated' in attributes:
            reasons.append("오염물 제거 후 배출해야 합니다")
        if 'contains_liquid' in attributes:
            reasons.append("내용물을 비우고 깨끗이 씻은 후 배출해야 합니다")
        if material == 'plastic_pp' and 'has_metal_cap' in attributes:
            reasons.append("금속 캡은 제거 후 배출해야 합니다")

        return ("재활용 불가", "\n".join(reasons)) if reasons else \
               ("재활용 가능", "색상, 파손 여부와 무관하게 재활용 가능합니다")

    if material == 'vinyl':
        if 'plastic_pp' in attributes:
            reasons.append("PP와 혼합된 비닐은 분리가 어려워 폐기 대상입니다")
        if 'contaminated' in attributes:
            reasons.append("오염물(음식물, 기름 등)은 제거 후 배출해야 합니다")
        if 'has_plastic_label' in attributes:
            reasons.append("플라스틱 라벨은 제거 후 배출해야 합니다")
        if 'broken' in attributes:
            reasons.append("손상된 비닐은 재활용이 어렵습니다")
        if 'contains_liquid' in attributes:
            reasons.append("내용물을 제거한 후 배출해야 합니다")

        return ("재활용 불가", "\n".join(reasons)) if reasons else \
               ("재활용 가능", "깨끗한 비닐은 재활용 가능합니다")

    if material == 'paper':
        if 'contaminated' in attributes:
            reasons.append("오염물 제거 후 배출해야 합니다")
        if 'paper_pack' in attributes:
            reasons.append("종이팩은 분리 가능 시 별도 배출해야 합니다")
        if 'plastic pp' in attributes:
            reasons.append("재질이 혼합된 종이는 일반쓰레기로 분류됩니다")

        return ("재활용 불가", "\n".join(reasons)) if reasons else \
               ("재활용 가능", "깨끗한 종이는 재활용 가능합니다")

    if material == 'paper_pack':
        if 'contaminated' in attributes:
            reasons.append("오염물 제거 후 배출해야 합니다")
        if 'has_plastic_cap' in attributes:
            reasons.append("플라스틱 뚜껑은 제거 후 배출해야 합니다")
        if 'plastic_pp' in attributes:
            reasons.append("재질이 혼합된 종이팩은 일반쓰레기로 분류됩니다")

        return ("재활용 불가", "\n".join(reasons)) if reasons else \
               ("재활용 가능", "분리 배출이 가능한 종이팩입니다")

    if material == 'styrofoam':
        if 'contaminated' in attributes:
            reasons.append("오염된 스티로폼은 일반쓰레기로 분류됩니다")
        if 'has_pattern' in attributes:
            reasons.append("패턴 있는 스티로폼은 일반쓰레기로 분류됩니다")

        return ("재활용 불가", "\n".join(reasons)) if reasons else \
               ("재활용 가능", "깨끗한 스티로폼은 재활용 가능합니다")

    if material == 'iron_can':
        if 'has_plastic_cap' in attributes:
            reasons.append("플라스틱 뚜껑은 제거 후 배출해야 합니다")
        if 'has_plastic_label' in attributes:
            reasons.append("플라스틱 라벨은 제거 후 배출해야 합니다")
        if 'contaminated' in attributes:
            reasons.append("내용물이나 기름 등으로 오염된 캔은 재활용이 어렵습니다")
        if 'contains_liquid' in attributes:
            reasons.append("내용물을 완전히 비운 후 배출해야 합니다")
        if 'has_metal_cap' in attributes:
            reasons.append("금속 캡은 지역에 따라 분리배출 기준이 다를 수 있습니다 (금속캔류 또는 고철류)")

        if reasons:
            return "재활용 불가", "\n".join(reasons)
        else:
            return "재활용 가능", "색상·디자인에 관계없이 재활용 가능합니다"

    if material == 'aluminium_can':
        if 'has_plastic_cap' in attributes:
            reasons.append("플라스틱 뚜껑은 제거 후 배출해야 합니다")
        if 'has_plastic_label' in attributes:
            reasons.append("플라스틱 라벨은 제거 후 배출해야 합니다")
        if 'contaminated' in attributes:
            reasons.append("내용물이나 기름 등으로 오염된 캔은 재활용이 어렵습니다")
        if 'contains_liquid' in attributes:
            reasons.append("내용물을 완전히 비운 후 배출해야 합니다")
        if 'has_metal_cap' in attributes:
            reasons.append("금속 캡은 지역에 따라 분리배출 기준이 다를 수 있습니다 (금속캔류 또는 고철류)")

        if reasons:
            return "재활용 불가", "\n".join(reasons)
        else:
            return "재활용 가능", "색상·디자인에 관계없이 재활용 가능합니다"

    if material == 'metal':
        if 'has_plastic_cap' in attributes:
            reasons.append("플라스틱 뚜껑은 제거 후 배출해야 합니다")
        if 'has_plastic_label' in attributes:
            reasons.append("플라스틱 라벨은 제거 후 배출해야 합니다")
        if 'contaminated' in attributes:
            reasons.append("오염물(음식물, 기름 등)은 제거 후 배출해야 합니다")
        if 'contains_liquid' in attributes:
            reasons.append("내용물을 완전히 비운 후 배출해야 합니다")
        if 'has_metal_cap' in attributes:
            reasons.append("금속 캡은 지역에 따라 금속캔류 또는 고철류로 배출해야 할 수 있습니다")

        if reasons:
            return "재활용 불가", "\n".join(reasons)
        else:
            return "재활용 가능", "색상·디자인 상관없이 금속 재질은 재활용 가능합니다"

    
    return "재활용 불가", "재질 또는 속성 조건이 재활용 기준을 충족하지 않습니다"


# ==================== UI 시작 ====================

st.title("♻️ 재활용 여부 판단 AI")

# 사이드바 입력
option = st.sidebar.selectbox("입력 방식 선택", ("이미지 업로드", "카메라 촬영"))

image = None
if option == "이미지 업로드":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif option == "카메라 촬영":
    camera_image = st.camera_input("카메라로 사진을 찍어주세요")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# ==================== 예측 ====================
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

        # 1차 재질 분류
        material_class = results[0].names[int(cls_ids[i])]

        # 2차 세부 재질 분류
        if material_class == 'plastic':
            with torch.no_grad():
                logits = resnet18_model(input_tensor)
                idx = torch.argmax(logits, dim=1).item()
                detailed_material = detailed_materials[idx]
        else:
            detailed_material = material_class

        # 3차 속성 분류
        attr_labels = run_attribute_classifier(input_tensor, detailed_material)

        # 재활용 여부 판단
        decision, reason = predict_rule(detailed_material, attr_labels)

        results_list.append((cropped, material_class, detailed_material, attr_labels, decision, reason))

    # ==================== 결과 출력 ====================
    st.subheader("🔍 분류 결과")
    for idx, (img, material, detailed, attrs, decision, reason) in enumerate(results_list):
        with st.expander(f"📦 객체 {idx+1} - {'✅' if '가능' in decision else '❌'} {decision}", expanded=False):
            st.image(img, caption=f"객체 {idx+1}", width=200)
            st.markdown(f"**1차 재질:** {material}")
            if material == 'plastic':
                st.markdown(f"**2차 세부 재질:** {detailed}")
            st.markdown(f"**속성:** {', '.join(attrs) if attrs else '없음'}")
            st.markdown(f"**판단 결과:** {'✅' if '가능' in decision else '❌'} {decision}")
            st.markdown(f"**사유:** {reason}")
