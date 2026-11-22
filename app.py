import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="HW1 | Прогноз цены автомобиля", layout="wide")

PICKLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw1_model.pickle")


@st.cache_resource
def load_artifact():
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


artifact = load_artifact()
pipeline = artifact["pipeline"]
train = artifact["train_data"]
feature_order = artifact["feature_order"]
medians = artifact["medians"]
metrics = artifact["metrics"]

st.title("Прогноз стоимости автомобиля")
st.markdown(
    "Модель: **Ridge** с one-hot кодированием категориальных признаков и "
    "стандартизацией. Обучена на 5840 объектах после удаления дубликатов. "
    f"Качество на тесте: R² = **{metrics['R2_test']:.3f}**, "
    f"доля прогнозов в коридоре ±10% = **{metrics['business_metric_test']:.1%}**."
)

tab_predict, tab_eda, tab_weights, tab_info = st.tabs(
    ["Предсказание", "EDA", "Веса модели", "О модели"]
)


def prepare_input(raw):
    data = raw.copy()
    for col in ("selling_price", "name", "torque"):
        if col in data.columns:
            data = data.drop(columns=[col])
    for col in ("mileage", "engine", "max_power"):
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.split().str[0], errors="coerce"
            )
    for col in data.columns:
        if data[col].isna().any() and col in medians:
            data[col] = data[col].fillna(medians[col])
    for col in ("engine", "seats"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(medians[col]).astype(int)
    missing = [col for col in feature_order if col not in data.columns]
    if missing:
        st.error(f"В файле отсутствуют признаки: {', '.join(missing)}")
        return None
    return data[feature_order]


def levels(col):
    return sorted(train[col].dropna().unique().tolist())


with tab_predict:
    mode = st.radio("Способ ввода", ["Один объект", "CSV-файл"], horizontal=True)

    if mode == "Один объект":
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.slider("Год выпуска", 2000, 2020, int(medians["year"]))
            km_driven = st.number_input("Пробег, км", 0, 500000,
                                        int(medians["km_driven"]), step=1000)
            mileage = st.number_input("Расход топлива", 0.0, 50.0,
                                      float(medians["mileage"]), step=0.1)
        with c2:
            engine = st.number_input("Объём двигателя, CC", 500, 5000,
                                     int(medians["engine"]), step=50)
            max_power = st.number_input("Мощность, bhp", 0.0, 500.0,
                                        float(medians["max_power"]), step=1.0)
            seats = st.selectbox("Число мест", levels("seats"),
                                 index=levels("seats").index(5)
                                 if 5 in levels("seats") else 0)
        with c3:
            fuel = st.selectbox("Топливо", levels("fuel"))
            seller_type = st.selectbox("Продавец", levels("seller_type"))
            transmission = st.selectbox("Коробка", levels("transmission"))
            owner = st.selectbox("Владелец", levels("owner"))

        if st.button("Предсказать цену", type="primary"):
            row = pd.DataFrame([{
                "year": year, "km_driven": km_driven, "mileage": mileage,
                "engine": engine, "max_power": max_power, "seats": seats,
                "fuel": fuel, "seller_type": seller_type,
                "transmission": transmission, "owner": owner,
            }])
            data = prepare_input(row)
            if data is not None:
                price = float(pipeline.predict(data)[0])
                st.success(f"Предсказанная цена: **{price:,.0f}**")
                st.caption(f"Ориентировочный коридор ±10%: "
                           f"{price * 0.9:,.0f} – {price * 1.1:,.0f}")
    else:
        uploaded = st.file_uploader("CSV с признаками автомобилей", type="csv")
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            st.caption(f"Загружено строк: {len(raw)}")
            data = prepare_input(raw)
            if data is not None and st.button("Предсказать цены", type="primary"):
                prices = pipeline.predict(data)
                result = data.copy()
                result["predicted_price"] = prices.round(0).astype("int64")
                st.dataframe(result.head(50), width="stretch")
                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button("Скачать предсказания", csv_out,
                                   "hw1_predictions.csv", "text/csv")

with tab_eda:
    st.subheader("Распределение целевой переменной")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train["selling_price"], bins=50, color="steelblue")
    axes[0].set_title("Цена (исходная шкала)")
    axes[1].hist(np.log1p(train["selling_price"]), bins=50, color="seagreen")
    axes[1].set_title("log(1 + цена)")
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Цена по типу топлива")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=train, x="fuel", y="selling_price", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.subheader("Мощность и цена")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=train.sample(1500, random_state=42), x="max_power",
                        y="selling_price", hue="fuel", alpha=0.6, ax=ax, s=18)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Корреляции числовых признаков")
    numeric = train.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Описательные статистики train"):
        st.dataframe(numeric.describe().round(2))

with tab_weights:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    weights = (pd.DataFrame({"feature": names, "weight": coefs})
               .assign(abs_weight=lambda d: d["weight"].abs())
               .sort_values("abs_weight", ascending=False)
               .reset_index(drop=True))
    st.subheader("Наиболее влиятельные признаки")
    fig, ax = plt.subplots(figsize=(9, 6))
    top = weights.head(20).iloc[::-1]
    ax.barh(top["feature"], top["weight"],
            color=np.where(top["weight"] >= 0, "seagreen", "indianred"))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент (изменение цены, INR)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Признаки стандартизованы: для числовых — изменение на 1 σ, "
               "для дамми — переход из базовой категории.")
    with st.expander("Все коэффициенты"):
        st.dataframe(weights.round(1), width="stretch")

with tab_info:
    st.subheader("Метрики зафиксированной модели")
    st.dataframe(pd.Series(metrics, name="value").to_frame().round(4))
    st.write("Лучшие гиперпараметры:", artifact["best_params"])
    st.write("CV R²:", round(artifact["cv_r2"], 4))
    st.write("Используемые признаки:", ", ".join(feature_order))
    st.markdown(
        "Ноутбук с полным решением: "
        "`homeworks/base/HW1_Regression_with_inference_base_solution.ipynb`. "
        "Модель сохранена в `hw1_model.pickle` вместе с пайплайном предобработки, "
        "медианами и обучающей выборкой для графиков."
    )
