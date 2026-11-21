import os
import re
import joblib
import shap
import numpy as np
import pandas as pd


BUILDING_TYPE_MAP = {
    "住宅大樓": "住宅大樓(11層含以上有電梯)",
    "華廈": "華廈(10層含以下有電梯)",
    "公寓": "公寓(5樓含以下無電梯)",
    "透天厝": "透天厝",
    "套房": "套房(1房1廳1衛)",
    "店面": "店面(店鋪)",
}

DISTRICT_ALIAS_MAP = {
    "臺東縣臺東市": "臺東縣台東市",
    "雲林縣臺西鄉": "雲林縣台西鄉",
}


class HousePricePredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "xgb_mc_house_price_model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ 找不到特徵檔：{feature_path}")

        self.model = joblib.load(model_path)
        self.model_features = list(joblib.load(feature_path))

        if isinstance(self.model, dict):
            self.base_model = self.model.get("base_model")
            self.model_low = self.model.get("model_low")
            self.model_high = self.model.get("model_high")
            self.q_score = (
                float(self.model["q_score"])
                if self.model.get("q_score") is not None
                else 0.0
            )
            self.alpha = (
                float(self.model["alpha"])
                if self.model.get("alpha") is not None
                else 0.1
            )
        else:
            self.base_model = self.model
            self.model_low = None
            self.model_high = None
            self.q_score = 0.0
            self.alpha = 0.1

        if self.base_model is None or not hasattr(self.base_model, "predict"):
            raise ValueError("❌ 模型檔中找不到可用的 base_model")

        self.explainer, self.base_value = self._init_shap_explainer(self.base_model)

    @staticmethod
    def _parse_base_value(expected_value):
        squeezed = np.squeeze(expected_value)
        if isinstance(squeezed, np.ndarray):
            squeezed = squeezed.item() if squeezed.size == 1 else squeezed.flat[0]
        if isinstance(squeezed, (bytes, str)):
            text = squeezed.decode("utf-8") if isinstance(squeezed, bytes) else squeezed
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if match:
                return float(match.group(0))
        return float(squeezed)

    def _init_shap_explainer(self, model):
        try:
            explainer = shap.TreeExplainer(model)
            base_val = self._parse_base_value(explainer.expected_value)
            return explainer, base_val
        except Exception:
            try:
                booster = model.get_booster() if hasattr(model, "get_booster") else model
                explainer = shap.TreeExplainer(booster)
                base_val = self._parse_base_value(explainer.expected_value)
                return explainer, base_val
            except Exception:
                return None, 0.0

    def _normalize_case(self, case_dict):
        normalized = dict(case_dict)
        if "district" in normalized and normalized["district"] in DISTRICT_ALIAS_MAP:
            normalized["district"] = DISTRICT_ALIAS_MAP[normalized["district"]]
        if "building_type" in normalized and normalized["building_type"] in BUILDING_TYPE_MAP:
            normalized["building_type"] = BUILDING_TYPE_MAP[normalized["building_type"]]
        return normalized

    def _align_features(self, case_dict):
        cleaned = self._normalize_case(case_dict)
        df = pd.DataFrame([cleaned])
        df = pd.get_dummies(df)

        missing_cols = {col: 0 for col in self.model_features if col not in df.columns}
        if missing_cols:
            missing_df = pd.DataFrame(missing_cols, index=df.index)
            df = pd.concat([df, missing_df], axis=1)

        aligned = df[self.model_features].copy()
        for col in aligned.columns:
            if aligned[col].dtype == "bool":
                aligned[col] = aligned[col].astype(int)
            elif pd.api.types.is_numeric_dtype(aligned[col]):
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)
        return aligned

    def _feature_to_human(self, feature, value):
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"
        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"
        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"

        val_float = float(value)
        human_map = {
            "building_area_sqm": f"建物總面積約 {val_float:.1f} 平方公尺（約 {val_float * 0.3025:.1f} 坪）",
            "main_area": f"主建物面積約 {val_float:.1f} 坪",
            "balcony_area": f"陽台面積約 {val_float:.1f} 坪",
            "building_age": f"屋齡約 {int(round(val_float))} 年",
            "floor": f"位於第 {int(round(val_float))} 樓",
            "total_floors": f"建物總樓層 {int(round(val_float))} 樓",
            "has_parking": "具備車位" if int(round(val_float)) == 1 else "未附車位",
            "has_elevator": "設有電梯" if int(round(val_float)) == 1 else "未設電梯",
        }
        return human_map.get(feature, feature)

    def predict(self, case_dict):
        X = self._align_features(case_dict)
        pred = float(self.base_model.predict(X)[0])

        lower_bound = None
        upper_bound = None
        half_width = None

        if self.model_low is not None and self.model_high is not None:
            y_low = float(self.model_low.predict(X)[0])
            y_high = float(self.model_high.predict(X)[0])
            adj_low = y_low + self.q_score
            adj_high = y_high - self.q_score
            lower_bound = max(0.0, min(adj_low, adj_high))
            upper_bound = max(0.0, max(adj_low, adj_high))
            half_width = (upper_bound - lower_bound) / 2.0

        explanation_lines = []
        if self.explainer is not None:
            try:
                shap_values = self.explainer(X)
                shap_vals = np.asarray(shap_values.values)[0]
                feature_names = X.columns
                base_value = self.base_value

                explanation_lines.append(
                    f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整："
                )

                idx_sorted = np.argsort(np.abs(shap_vals))[::-1]
                shown_count = 0
                for i in idx_sorted:
                    feat = feature_names[i]
                    val = X.iloc[0][feat]
                    shap_val = float(shap_vals[i])
                    if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                        continue
                    human_text = self._feature_to_human(feat, val)
                    direction = "推升" if shap_val > 0 else "下修"
                    explanation_lines.append(
                        f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪"
                    )
                    shown_count += 1
                    if shown_count >= 10:
                        break
            except Exception:
                explanation_lines.append("📌 目前環境無法產生 SHAP 細項拆解，直接顯示模型預測結果：")
        else:
            explanation_lines.append("📌 目前環境無法產生 SHAP 細項拆解，直接顯示模型預測結果：")

        explanation_lines.append(
            f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。"
        )

        confidence_pct = int(round((1 - self.alpha) * 100))
        if lower_bound is not None and upper_bound is not None:
            explanation_lines.append(
                f"📊 {confidence_pct}% 預測區間（CQR）："
                f"約 {lower_bound:.1f} ～ {upper_bound:.1f} 萬 / 坪（q_score = {self.q_score:.4f}）。"
            )
        else:
            explanation_lines.append(f"📊 {confidence_pct}% 預測區間：模型檔中未包含上下分位數模型。")

        return {
            "predicted_price": pred,
            "ci_90_lower": lower_bound,
            "ci_90_upper": upper_bound,
            "ci_90_radius": half_width,
            "confidence_pct": confidence_pct,
            "explanation": "\n".join(explanation_lines),
        }