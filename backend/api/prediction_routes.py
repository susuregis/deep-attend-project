import io
from datetime import datetime
from typing import Optional

import torch
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from sqlalchemy.orm import Session

from database import AttentionMetric, ClassSession, get_db

router = APIRouter(tags=["Predição"])


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    room_code: Optional[str] = None,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

    app_state = request.app.state
    model = app_state.model
    transform = app_state.transform
    config = app_state.model_config
    device = app_state.device

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred_class = prob.argmax(1).item()
        confidence = prob[0][pred_class].item()
        predicted_label = config["classes"][pred_class]

    prob_attentive = float(prob[0][0])
    prob_inattentive = (
        float(prob[0][1]) if len(prob[0]) > 1 else 1 - prob_attentive
    )

    attentive_label = config.get("attentive_label")
    if attentive_label:
        atento = str(predicted_label).lower() == str(attentive_label).lower()
    else:
        atento = (pred_class == 0) or ("aten" in str(predicted_label).lower())

    if room_code and user_id:
        session = db.query(ClassSession).filter(ClassSession.room_code == room_code).first()
        if session:
            metric = AttentionMetric(
                user_id=user_id,
                session_id=session.id,
                is_attentive=bool(atento),
                confidence=confidence,
                prob_attentive=prob_attentive,
                prob_inattentive=prob_inattentive,
            )
            db.add(metric)
            db.commit()

    app_state.request_count += 1

    return JSONResponse(
        content={
            "success": True,
            "atento": bool(atento),
            "classe": predicted_label,
            "pred_class_index": int(pred_class),
            "confianca": round(float(confidence), 4),
            "probabilidades": {
                config["classes"][i]: round(float(prob[0][i]), 4)
                for i in range(len(config["classes"]))
            },
            "percentuais": {
                config["classes"][i]: f"{round(float(prob[0][i]) * 100, 1)}%"
                for i in range(len(config["classes"]))
            },
            "prob_atento": round(prob_attentive * 100, 1),
            "prob_desatento": round(prob_inattentive * 100, 1),
            "timestamp": datetime.now().isoformat(),
        }
    )

