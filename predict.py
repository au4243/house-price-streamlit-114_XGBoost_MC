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

        self.raw_obj = joblib.load(model_path)
        self.model_features = list(joblib.load(feature_path))

        self.base_model = None
        self.mc_models = []
        self.model_low = None
        self.model_high = None
        self.q_score = 0.0
        self.alpha = 0.10
        self.z_value = 1.6448536269514722  # 雙尾 90% 信賴區間 z 值

        self.fixed_std = None
        self.fixed_radius = None
        self.residuals = None
        self.sample_preds = None
        self.interval_source = "未設定"

        self._parse_model_object(self.raw_obj)

        if self.base_model is None or not hasattr(self.base_model, "predict"):
            raise ValueError("❌ 模型檔中找不到可用的主預測模型（base_model）")

        self.explainer, self.base_value = self._init_shap_explainer(self.base_model)

    @staticmethod
    def _is_predictor(obj):
        return obj is not None and hasattr(obj, "predict") and callable(obj.predict)

    @staticmethod
    def _to_float_or_none(val):
        if val is None:
            return None
        try:
            arr = np.asarray(val, dtype=float)
            if arr.size == 1:
                num = float(arr.item())
                return num if np.isfinite(num) else None
        except Exception:
            pass
        return None

    def _parse_model_object(self, obj):
        if isinstance(obj, dict):
            # 1. 主模型
            for key in ["base_model", "model", "mean_model", "estimator", "xgb_model"]:
                if key in obj and self._is_predictor(obj[key]):
                    self.base_model = obj[key]
                    break

            # 2. 信心水準 / alpha / z 值
            if "alpha" in obj and self._to_float_or_none(obj["alpha"]) is not None:
                a = self._to_float_or_none(obj["alpha"])
                if 0.0 < a < 1.0:
                    self.alpha = a
                    self.z_value = 1.6448536269514722 if abs(a - 0.10) < 1e-6 else 1.959963984540054

            if "z_score" in obj and self._to_float_or_none(obj["z_score"]) is not None:
                self.z_value = self._to_float_or_none(obj["z_score"])

            if "q_score" in obj and self._to_float_or_none(obj["q_score"]) is not None:
                self.q_score = self._to_float_or_none(obj["q_score"])

            # 3. Monte Carlo 多模型集合
            for key in ["mc_models", "models", "estimators", "ensemble_models", "bootstrap_models"]:
                if key in obj and isinstance(obj[key], (list, tuple)):
                    valid_models = [m for m in obj[key] if self._is_predictor(m)]
                    if valid_models:
                        self.mc_models = valid_models
                        if self.base_model is None:
                            self.base_model = valid_models[0]
                        break

            # 4. 分位數模型（若同時存在）
            if self._is_predictor(obj.get("model_low")) and self._is_predictor(obj.get("model_high")):
                self.model_low = obj["model_low"]
                self.model_high = obj["model_high"]

            # 5. 數值型不確定度 / 標準差 / 半徑 / 殘差 / 抽樣預測
            for key in ["std", "sigma", "mc_std", "pred_std", "uncertainty", "rmse", "mae"]:
                if key in obj:
                    val = self._to_float_or_none(obj[key])
                    if val is not None and val >= 0:
                        self.fixed_std = val
                        self.interval_source = f"模型標準差（{key}）"
                        break

            for key in ["radius", "margin", "ci_radius", "half_width", "q90", "q_90", "interval_half_width"]:
                if key in obj:
                    val = self._to_float_or_none(obj[key])
                    if val is not None and val >= 0:
                        self.fixed_radius = val
                        self.interval_source = f"區間半徑（{key}）"
                        break

            for key in ["residuals", "errors", "val_residuals", "test_residuals", "abs_errors"]:
                if key in obj and obj[key] is not None:
                    try:
                        arr = np.asarray(obj[key], dtype=float).ravel()
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            self.residuals = arr
                            self.interval_source = f"殘差分布（{key}）"
                            break
                    except Exception:
                        pass

            for key in ["mc_preds", "mc_samples", "samples", "preds_samples", "posterior_samples"]:
                if key in obj and obj[key] is not None:
                    try:
                        arr = np.asarray(obj[key], dtype=float)
                        if arr.ndim == 1 and arr.size >= 2 and np.all(np.isfinite(arr)):
                            self.sample_preds = arr
                            self.interval_source = f"抽樣預測分佈（{key}）"
                            break
                    except Exception:
                        pass

            if self.mc_models:
                self.interval_source = f"MC 集成模型（共 {len(self.mc_models)} 組）"
            elif self.model_low is not None and self.model_high is not None:
                self.interval_source = "CQR 分位數模型"

        elif isinstance(obj, (list, tuple)):
            valid_models = [m for m in obj if self._is_predictor(m)]
            if valid_models:
                self.mc_models = valid_models
                self.base_model = valid_models[0]
                self.interval_source = f"MC 模型清單（共 {len(valid_models)} 組）"
        elif self._is_predictor(obj):
            self.base_model = obj

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

    def _compute_90_interval(self, X, pred):
        # 情況 1：MC 多模型集成，每組模型對單筆樣本預測一個值
        if self.mc_models:
            mc_vals = np.array([float(m.predict(X)[0]) for m in self.mc_models], dtype=float)
            mc_vals = mc_vals[np.isfinite(mc_vals)]
            if mc_vals.size >= 2:
                lower = max(0.0, float(np.percentile(mc_vals, 5)))
                upper = max(lower, float(np.percentile(mc_vals, 95)))
                radius = (upper - lower) / 2.0
                return lower, upper, radius, float(np.mean(mc_vals))

        # 情況 2：模型檔內含固定抽樣預測分佈
        if self.sample_preds is not None and self.sample_preds.size >= 2:
            lower = max(0.0, float(np.percentile(self.sample_preds, 5)))
            upper = max(lower, float(np.percentile(self.sample_preds, 95)))
            radius = (upper - lower) / 2.0
            return lower, upper, radius, pred

        # 情況 3：CQR 上下分位數模型
        if self.model_low is not None and self.model_high is not None:
            y_low = float(self.model_low.predict(X)[0])
            y_high = float(self.model_high.predict(X)[0])
            adj_low = y_low + self.q_score
            adj_high = y_high - self.q_score
            lower = max(0.0, min(adj_low, adj_high))
            upper = max(lower, max(adj_low, adj_high))
            radius = (upper - lower) / 2.0
            return lower, upper, radius, pred

        # 情況 4：模型檔內含區間半徑
        if self.fixed_radius is not None:
            lower = max(0.0, pred - self.fixed_radius)
            upper = max(lower, pred + self.fixed_radius)
            radius = (upper - lower) / 2.0
            return lower, upper, radius, pred

        # 情況 5：模型檔內含標準差 / 不確定度
        if self.fixed_std is not None:
            radius = self.z_value * self.fixed_std
            lower = max(0.0, pred - radius)
            upper = max(lower, pred + radius)
            return lower, upper, (upper - lower) / 2.0, pred

        # 情況 6：模型檔內含殘差陣列，以第 90 百分位絕對誤差作為半徑
        if self.residuals is not None and self.residuals.size > 0:
            radius = float(np.percentile(np.abs(self.residuals), 90))
            lower = max(0.0, pred - radius)
            upper = max(lower, pred + radius)
            return lower, upper, (upper - lower) / 2.0, pred

        # 情況 7：若模型檔僅存單一模型且無任何不確定度參數，提供保守基準區間（±10%）以避免空白
        radius = max(0.5, pred * 0.10)
        self.interval_source = "預設 ±10% 估算區間（模型檔未附帶區間參數）"
        lower = max(0.0, pred - radius)
        upper = max(lower, pred + radius)
        return lower, upper, (upper - lower) / 2.0, pred

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
        raw_pred = float(self.base_model.predict(X)[0])

        lower_bound, upper_bound, half_width, pred = self._compute_90_interval(X, raw_pred)

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
        explanation_lines.append(
            f"📊 90% 預測區間：約 {lower_bound:.1f} ～ {upper_bound:.1f} 萬 / 坪（±{half_width:.1f} 萬 / 坪，來源：{self.interval_source}）。"
        )

        return {
            "predicted_price": pred,
            "ci_90_lower": lower_bound,
            "ci_90_upper": upper_bound,
            "ci_90_radius": half_width,
            "confidence_pct": 90,
            "interval_source": self.interval_source,
            "explanation": "\n".join(explanation_lines),
        }